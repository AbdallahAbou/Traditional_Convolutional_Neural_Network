"""
Data Ingestion Script
Loads patients.csv into MySQL and indexes documents into ChromaDB
Uses OpenAI text-embedding-3-large for embeddings
"""

import os
import sys
import csv
import re
import glob
import time
import logging
from pathlib import Path
import mysql.connector
import chromadb
from chromadb.config import Settings
import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3307)),
    "user": "root",
    "password": os.getenv("MYSQL_ROOT_PASSWORD", "ehrRoot2024!"),
    "database": "ehr_demo"
}

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8100))

DATA_DIR = Path(__file__).parent.parent
PATIENTS_CSV = DATA_DIR / "patients.csv"
DOCS_DIR = DATA_DIR / "docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


class EmbeddingAPI:
    """OpenAI Embeddings API client using text-embedding-3-large"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.model_name = "text-embedding-3-large"
        self.dimension = 3072
        logger.info(f"Initialized OpenAI embeddings: {self.model_name} ({self.dimension} dims)")
    
    def embed(self, texts):
        """Generate embeddings via OpenAI API"""
        if not texts:
            return []
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "input": texts
                }
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]


def connect_mysql():
    logger.info(f"Connecting to MySQL at {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}...")
    return mysql.connector.connect(**MYSQL_CONFIG)


def connect_chroma():
    logger.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    return chromadb.HttpClient(
        host=CHROMA_HOST, 
        port=CHROMA_PORT, 
        settings=Settings(anonymized_telemetry=False)
    )


def load_patients(conn):
    """Load patients from CSV into MySQL"""
    cursor = conn.cursor()
    logger.info(f"Loading patients from {PATIENTS_CSV}...")
    
    with open(PATIENTS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            cursor.execute("""
                INSERT INTO patients (
                    patient_id, age, sex, height_cm, weight_kg, bmi, smoking_status,
                    primary_diagnosis, disease_stage, secondary_conditions, icd10_codes,
                    bp_systolic, bp_diastolic, heart_rate, temperature_c,
                    alt_u_l, ast_u_l, bilirubin_mg_dl, afp_ng_ml,
                    creatinine_mg_dl, egfr_ml_min, inr, platelets_k_ul,
                    allergies, contraindications, current_medications,
                    recent_imaging_type, recent_imaging_date, key_imaging_findings,
                    procedures, admission_date, discharge_date, follow_up_date, follow_up_plan
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE age=VALUES(age), primary_diagnosis=VALUES(primary_diagnosis)
            """, (
                row.get('patient_id'),
                int(row.get('age', 0)) if row.get('age') else None,
                row.get('sex'),
                float(row.get('height_cm', 0)) if row.get('height_cm') else None,
                float(row.get('weight_kg', 0)) if row.get('weight_kg') else None,
                float(row.get('bmi', 0)) if row.get('bmi') else None,
                row.get('smoking_status', 'never'),
                row.get('primary_diagnosis'),
                row.get('disease_stage') if row.get('disease_stage') != 'N/A' else None,
                row.get('secondary_conditions'),
                row.get('icd10_codes'),
                int(row.get('bp_sys', 0)) if row.get('bp_sys') else None,
                int(row.get('bp_dia', 0)) if row.get('bp_dia') else None,
                int(row.get('heart_rate', 0)) if row.get('heart_rate') else None,
                float(row.get('temperature_c', 0)) if row.get('temperature_c') else None,
                int(row.get('alt_u_l', 0)) if row.get('alt_u_l') else None,
                int(row.get('ast_u_l', 0)) if row.get('ast_u_l') else None,
                float(row.get('bilirubin_mg_dl', 0)) if row.get('bilirubin_mg_dl') else None,
                float(row.get('afp_ng_ml', 0)) if row.get('afp_ng_ml') else None,
                float(row.get('creatinine_mg_dl', 0)) if row.get('creatinine_mg_dl') else None,
                float(row.get('egfr_ml_min_1_73m2', 0)) if row.get('egfr_ml_min_1_73m2') else None,
                float(row.get('inr', 0)) if row.get('inr') else None,
                int(row.get('platelets_k_ul', 0)) if row.get('platelets_k_ul') else None,
                row.get('allergies'),
                row.get('contraindications'),
                row.get('current_medications'),
                row.get('recent_imaging_type'),
                row.get('recent_imaging_date') if row.get('recent_imaging_date') else None,
                row.get('key_imaging_findings'),
                row.get('procedures'),
                row.get('admission_date') if row.get('admission_date') else None,
                row.get('discharge_date') if row.get('discharge_date') else None,
                row.get('follow_up_date') if row.get('follow_up_date') else None,
                row.get('follow_up_plan')
            ))
            count += 1
            if count % 20 == 0:
                logger.info(f"  Loaded {count} patients...")
        conn.commit()
        logger.info(f"Loaded {count} patients into MySQL")
    cursor.close()
    return count


def chunk_document(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Section-aware chunking for clinical documents"""
    text = text.strip()
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    current_section = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Detect section headers
        if re.match(r'^#+\s+', para) or re.match(r'^[A-Z][A-Za-z\s]+:$', para):
            current_section = para.strip('#: ')
        
        # Check if adding this paragraph exceeds chunk size
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append({"text": current_chunk.strip(), "section": current_section})
            # Create overlap from end of previous chunk
            words = current_chunk.split()
            overlap_words = words[-overlap//5:] if len(words) > overlap//5 else []
            current_chunk = ' '.join(overlap_words) + '\n\n'
        
        current_chunk += para + '\n\n'
    
    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "section": current_section})
    
    return chunks


def load_documents(conn, chroma_client, embedder):
    """Load and index clinical documents"""
    cursor = conn.cursor()
    
    # Clear existing collection
    try:
        chroma_client.delete_collection("ehr_chunks")
        logger.info("Cleared existing collection")
    except Exception:
        pass
    
    collection = chroma_client.create_collection(
        name="ehr_chunks", 
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info(f"Processing documents from {DOCS_DIR}...")
    doc_files = sorted(glob.glob(str(DOCS_DIR / "*.md")))
    total_chunks = 0
    batch_size = 10  # Smaller batches for large embeddings
    
    all_chunks = []
    
    for doc_path in doc_files:
        filename = os.path.basename(doc_path)
        match = re.match(r'(P-\d+)_(\w+)\.md', filename)
        if not match:
            continue
        
        patient_id = match.group(1)
        doc_type = match.group(2)
        doc_id = f"{patient_id}_{doc_type}"
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        word_count = len(content.split())
        date_match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', content)
        doc_date = date_match.group(1) if date_match else None
        
        # Insert document record
        cursor.execute("""
            INSERT INTO documents (doc_id, patient_id, doc_type, title, file_path, doc_date, word_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE word_count=VALUES(word_count)
        """, (doc_id, patient_id, doc_type, f"{doc_type.title()} Summary", doc_path, doc_date, word_count))
        
        # Chunk document
        chunks = chunk_document(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "patient_id": patient_id,
                "doc_type": doc_type,
                "section": chunk.get("section", ""),
                "text": chunk["text"],
                "chunk_index": i
            })
            
            # Insert chunk record
            cursor.execute("""
                INSERT INTO chunks (chunk_id, doc_id, patient_id, chunk_index, section_header, word_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE section_header=VALUES(section_header)
            """, (chunk_id, doc_id, patient_id, i, chunk.get("section", ""), len(chunk["text"].split())))
        
        total_chunks += len(chunks)
    
    conn.commit()
    cursor.close()
    
    # Index chunks in batches
    logger.info(f"Embedding {len(all_chunks)} chunks via OpenAI API...")
    start_time = time.time()
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [{
            "doc_id": c["doc_id"], 
            "patient_id": c["patient_id"], 
            "doc_type": c["doc_type"], 
            "section": c["section"], 
            "chunk_index": c["chunk_index"]
        } for c in batch]
        
        embeddings = embedder.embed(texts)
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        
        elapsed = time.time() - start_time
        logger.info(f"  Batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1} - {elapsed:.1f}s elapsed")
    
    total_time = time.time() - start_time
    logger.info(f"Indexed {len(doc_files)} documents with {total_chunks} chunks in {total_time:.1f}s")
    return len(doc_files), total_chunks


def main():
    logger.info("=" * 60)
    logger.info("EHR RAG Demo - Data Ingestion")
    logger.info("=" * 60)
    
    # Initialize embedder
    try:
        embedder = EmbeddingAPI()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Connect to MySQL
    try:
        mysql_conn = connect_mysql()
        logger.info("Connected to MySQL")
    except Exception as e:
        logger.error(f"MySQL connection failed: {e}")
        sys.exit(1)
    
    # Connect to ChromaDB
    try:
        chroma_client = connect_chroma()
        chroma_client.heartbeat()
        logger.info("Connected to ChromaDB")
    except Exception as e:
        logger.error(f"ChromaDB connection failed: {e}")
        mysql_conn.close()
        sys.exit(1)
    
    # Run ingestion
    try:
        patient_count = load_patients(mysql_conn)
        doc_count, chunk_count = load_documents(mysql_conn, chroma_client, embedder)
        
        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Patients: {patient_count}")
        logger.info(f"  Documents: {doc_count}")
        logger.info(f"  Chunks indexed: {chunk_count}")
        logger.info(f"  Embedding model: text-embedding-3-large (3072 dims)")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        mysql_conn.close()


if __name__ == "__main__":
    main()
