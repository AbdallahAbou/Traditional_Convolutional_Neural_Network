# Building an Agentic RAG System for Electronic Health Records

> A technical deep-dive into building a production-grade AI assistant that operates over synthetic patient data, combining structured database queries with semantic retrieval over clinical notes.

---

## Introduction

This project demonstrates how to build an **agentic RAG (Retrieval-Augmented Generation) system** specifically designed for healthcare data. Unlike simple chatbots, this system operates as a constrained planner that:

1. Interprets clinical queries
2. Selects appropriate data retrieval tools
3. Executes multi-stage retrieval pipelines
4. Synthesizes answers grounded in evidence
5. Provides full observability into its reasoning

The architecture prioritizes **security boundaries**, **retrieval correctness**, and **auditability**—principles that matter in any domain handling sensitive data, but are non-negotiable in healthcare contexts.

---

## Why This Architecture?

### The Problem with Naive RAG

A typical RAG implementation embeds documents, performs vector search, and passes results to an LLM. This fails in clinical settings because:

- **Heterogeneous data**: Patient records combine structured data (labs, vitals, medications) with unstructured narratives (physician notes, imaging reports)
- **Scope isolation**: Queries must be strictly bounded to a single patient—cross-patient data leakage is unacceptable
- **Precision requirements**: Medical queries demand high retrieval precision; false positives can mislead clinical reasoning
- **Auditability**: Every inference must be traceable to source evidence

### Our Approach: Tool-Driven Agentic RAG

Instead of a monolithic retrieval pipeline, we decompose the system into:

1. **Structured data tools**: Direct SQL queries for discrete facts (labs, medications, vitals)
2. **Semantic retrieval tools**: Multi-stage vector search for unstructured notes
3. **An LLM planner**: Orchestrates tool selection based on query intent

The LLM never accesses data directly. It can only request data through validated tool interfaces, with patient scope enforced server-side.

---

## System Architecture

### Component Overview

```
+------------------+       +-------------------+       +------------------+
|    Frontend      | ----> |   FastAPI Backend | ----> |     MySQL        |
|  (React + Vite)  |       |   (Agent + Tools) |       |  (Structured)    |
+------------------+       +-------------------+       +------------------+
                                    |
                                    v
                           +-------------------+       +------------------+
                           |   OpenAI APIs     | <---> |    ChromaDB      |
                           | (LLM + Embeddings |       |   (Vector DB)    |
                           |   + Reranking)    |       +------------------+
                           +-------------------+
```

### Technology Selection Rationale

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Database** | MySQL 8.0 | ACID compliance, mature ecosystem, strong permission model |
| **Vector Store** | ChromaDB | Lightweight, metadata filtering, easy deployment |
| **Backend** | FastAPI | Async support, automatic OpenAPI, Pydantic validation |
| **LLM** | GPT-4o | Best-in-class tool calling, medical knowledge |
| **Embeddings** | text-embedding-3-large | 3072 dimensions, superior semantic fidelity |
| **Reranking** | OpenAI Rerank API | Cross-encoder precision for medical terminology |
| **Frontend** | React + Vite | Fast iteration, component model suits trace visualization |

---

## Data Model

### Structured Data (MySQL)

The `patients` table consolidates demographics, vitals, laboratory values, medications, and care plans into a single denormalized record. This design choice optimizes for read patterns typical in clinical queries:

```sql
CREATE TABLE patients (
    patient_id VARCHAR(20) PRIMARY KEY,  -- Pseudonymous identifier (P-1001)
    
    -- Demographics
    age INT,
    sex ENUM('M', 'F'),
    height_cm DECIMAL(5,2),
    weight_kg DECIMAL(5,2),
    bmi DECIMAL(4,1),
    smoking_status ENUM('never', 'former', 'current'),
    
    -- Clinical
    primary_diagnosis VARCHAR(255),
    disease_stage VARCHAR(50),
    secondary_conditions TEXT,
    icd10_codes VARCHAR(255),
    
    -- Vitals
    bp_systolic INT,
    bp_diastolic INT,
    heart_rate INT,
    temperature_c DECIMAL(4,2),
    
    -- Laboratory Values
    alt_u_l INT,
    ast_u_l INT,
    creatinine_mg_dl DECIMAL(4,2),
    egfr_ml_min INT,
    
    -- Medications & Safety
    allergies TEXT,
    contraindications TEXT,
    current_medications TEXT,
    
    -- Care Plan
    follow_up_date DATE,
    follow_up_plan TEXT
);
```

**Security enforcement**: A dedicated `ehr_agent` user has `SELECT`-only permissions. No tool can execute `INSERT`, `UPDATE`, or `DELETE` operations.

### Unstructured Data (Vector Store)

Clinical notes are stored as embedded chunks in ChromaDB:

```
Collection: ehr_chunks

Document structure:
- chunk_id: Unique identifier (e.g., "P-1001_discharge_chunk_2")
- embedding: 3072-dimensional vector (text-embedding-3-large)
- text: Raw chunk content
- metadata:
    - patient_id: Scope filter (mandatory)
    - doc_id: Parent document reference
    - doc_type: Note category (discharge, progress, imaging)
    - section: Document section (Assessment, Plan, Labs)
    - chunk_index: Position within document
```

**Critical constraint**: Every vector query includes a mandatory `patient_id` metadata filter. The system cannot perform cross-patient semantic search.

---

## Retrieval Pipeline

Medical retrieval demands higher precision than general-domain applications. We implement a three-stage pipeline:

### Stage 1: Candidate Recall (Dense Retrieval)

```python
# Embed query using text-embedding-3-large
query_embedding = embed(query)

# Retrieve k1 candidates with mandatory patient filter
candidates = collection.query(
    query_embeddings=[query_embedding],
    n_results=50,  # High recall
    where={"patient_id": patient_id}  # Scope enforcement
)
```

**Goal**: Maximize recall. Accept noise—we filter in subsequent stages.

**Embedding model**: `text-embedding-3-large` (3072 dimensions) provides superior semantic representation for medical terminology compared to smaller models.

### Stage 2: Reranking (Cross-Encoder)

Dense retrieval optimizes for embedding similarity, which can miss semantic nuance. Reranking applies a cross-encoder that jointly attends to query and document:

```python
# Rerank candidates using OpenAI Rerank API
reranked = openai.rerank(
    model="rerank-1",
    query=query,
    documents=[c["text"] for c in candidates],
    top_n=10  # Precision-focused
)
```

**Why reranking matters**: A query like "contraindications for sorafenib" might retrieve chunks mentioning the drug name but not contraindication context. Cross-encoder reranking evaluates semantic relevance holistically.

### Stage 3: Evidence Selection

Final filtering applies domain logic:

1. **Deduplication**: Remove near-duplicate chunks
2. **Section prioritization**: Imaging findings > Laboratory results > Assessment > General notes
3. **Recency weighting**: Recent notes preferred for time-sensitive queries

---

## Tool Architecture

The LLM interacts with data exclusively through a validated tool interface.

### Structured Data Tools

```python
@tool_registry.register
def get_patient_profile(patient_id: str) -> dict:
    """Retrieve demographics, diagnosis, and vitals."""
    # patient_id injected server-side, not from LLM
    return db.query("""
        SELECT patient_id, age, sex, primary_diagnosis, 
               disease_stage, bp_systolic, bp_diastolic
        FROM patients WHERE patient_id = %s
    """, [patient_id])

@tool_registry.register
def get_latest_labs(patient_id: str) -> dict:
    """Retrieve most recent laboratory values."""
    return db.query("""
        SELECT alt_u_l, ast_u_l, creatinine_mg_dl, 
               egfr_ml_min, inr, platelets_k_ul
        FROM patients WHERE patient_id = %s
    """, [patient_id])

@tool_registry.register
def get_medications(patient_id: str) -> dict:
    """Retrieve current medication list."""
    # Returns structured medication data
    
@tool_registry.register
def get_safety_info(patient_id: str) -> dict:
    """Retrieve allergies and contraindications."""
    # Critical for drug interaction queries
```

### Retrieval Tools

```python
@tool_registry.register
def retrieve_note_chunks(patient_id: str, query: str, top_k: int = 10) -> list:
    """
    Multi-stage retrieval over clinical notes.
    
    Pipeline:
    1. Dense retrieval (k=50)
    2. Cross-encoder reranking (top_k)
    3. Evidence selection
    """
    candidates = vector_store.search(patient_id, query, k=50)
    reranked = reranker.rerank(query, candidates, top_n=top_k)
    return format_evidence(reranked)
```

### Security Properties

1. **No raw SQL**: Tools execute parameterized queries only
2. **Patient scope injection**: `patient_id` comes from validated request context, not LLM output
3. **Read-only access**: Database user cannot modify data
4. **Row limits**: All queries enforce result limits

---

## Agent Design

The LLM operates as a constrained planner, not an autonomous agent.

### System Prompt (Abbreviated)

```
You are a clinical data assistant operating over synthetic EHR records.

CONSTRAINTS:
- You can ONLY access data through provided tools
- You CANNOT select or change the patient context
- You MUST cite evidence for every clinical claim
- You MUST NOT provide diagnostic or treatment recommendations
- You MUST acknowledge when evidence is insufficient

AVAILABLE TOOLS:
- get_patient_profile: Demographics and vitals
- get_latest_labs: Laboratory values
- get_medications: Current prescriptions
- get_safety_info: Allergies and contraindications
- retrieve_note_chunks: Search clinical notes

For each query:
1. Identify what data sources are needed
2. Call appropriate tools
3. Synthesize findings with citations
4. State confidence level
```

### Planning Behavior

The agent decomposes compound queries:

**Query**: "What are the contraindications for starting this patient on anticoagulation?"

**Agent plan**:
1. `get_medications()` — Current drugs (interaction check)
2. `get_safety_info()` — Known contraindications
3. `get_latest_labs()` — Coagulation markers (INR, platelets)
4. `retrieve_note_chunks("bleeding risk factors")` — Clinical notes

---

## Security Model

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Cross-patient data access | Mandatory patient_id filter on all queries |
| SQL injection | Parameterized queries only, no dynamic SQL |
| Prompt injection | System prompt priority, document content isolated |
| Data exfiltration | No write access, audit logging |
| LLM hallucination | Grounding requirement, citation enforcement |

### Patient Scope Isolation

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    # Validate patient_id from request (not from query content)
    patient_id = validate_patient_id(request.patient_id)
    
    # Inject into tool context—LLM cannot override
    tool_context = ToolContext(patient_id=patient_id)
    
    # Agent can only access this patient's data
    response = await agent.process(
        query=request.query,
        context=tool_context
    )
```

### Prompt Injection Defense

Retrieved documents are treated as untrusted input:

```python
def format_tool_result(tool_name: str, result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result)  # Serialized, not interpolated
    }
```

The LLM receives tool outputs as structured data, not as part of the instruction stream.

---

## Observability

Every request generates a complete audit trail:

```sql
CREATE TABLE audit_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(36),      -- Groups related events
    patient_id VARCHAR(20),
    event_type ENUM('query', 'tool_call', 'retrieval', 'response', 'error'),
    payload JSON,                -- Tool inputs, outputs, scores
    latency_ms INT,
    tokens_used INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Trace Panel (Frontend)

The UI exposes:
- Tool call sequence with timing
- Retrieved chunk IDs and relevance scores
- Token consumption breakdown
- End-to-end latency

This transparency serves two purposes:
1. **Debugging**: Understand why the agent selected specific evidence
2. **Trust**: Users can verify claims against source documents

---

## Safety Posture

### Medical Disclaimer

This system operates on **synthetic data only**. Real patient data would require:
- HIPAA-compliant infrastructure
- BAA agreements with cloud providers
- PHI encryption at rest and in transit
- Access audit requirements
- Institutional review board approval

### Behavioral Constraints

The system is explicitly prohibited from:
- Providing diagnostic conclusions
- Recommending treatment changes
- Suggesting medication dosages
- Making prognostic statements

Output language is restricted to **descriptive analysis**: "The records indicate..." rather than "The patient should..."

---

## Deployment Architecture

### Docker Composition

```yaml
services:
  mysql:
    image: mysql:8.0
    # Read-only user for agent, root for ingestion
    
  chromadb:
    image: chromadb/chroma:latest
    # Persistent volume for embeddings
    
  backend:
    build: ./Dockerfile.backend
    # FastAPI + Agent + Tools
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    
  frontend:
    build: ./Dockerfile.frontend
    # React SPA, no secrets
```

### Network Isolation

- `ehr-net`: Internal communication (backend <-> databases)
- `global-net`: External access via Traefik reverse proxy
- Frontend has no direct database access

---

## Performance Considerations

### Latency Budget

| Stage | Target | Notes |
|-------|--------|-------|
| Embedding | 100ms | Single API call |
| Vector search | 50ms | ChromaDB local |
| Reranking | 500ms | Bottleneck—batch optimization |
| LLM generation | 1-3s | Depends on tool calls |

### Token Management

- Chunk size capped at 500 tokens
- Retrieved evidence limited to k=10 chunks
- Summaries preferred over raw text for context
- Total context budget: ~8K tokens

---

## Conclusion

This architecture demonstrates that building AI systems for sensitive domains requires more than prompt engineering. It requires:

1. **Explicit security boundaries**: The LLM operates in a sandbox with no direct data access
2. **Multi-stage retrieval**: Dense search alone is insufficient for medical precision
3. **Tool abstraction**: Validated interfaces enforce constraints the LLM cannot circumvent
4. **Full observability**: Every inference is auditable

The same patterns apply beyond healthcare—financial services, legal document review, and any domain where correctness and auditability outweigh generative creativity.

---

## Implementation Summary

### Final System Stats

| Metric | Value |
|--------|-------|
| Patients | 100 |
| Clinical Documents | 300 |
| Indexed Chunks | 793 |
| Embedding Dimensions | 3072 |
| Avg Retrieval Latency | ~300ms |
| Avg LLM Latency | ~5s |

### Service Endpoints

| Service | Port | Purpose |
|---------|------|---------|
| MySQL | 3307 | Structured patient data |
| ChromaDB | 8100 | Vector storage |
| Backend API | 8200 | Agent orchestration |
| Frontend | 3100 | React SPA |

### API Endpoints

```
GET  /health              - Service health check
GET  /patients            - List all patients
GET  /patients/{id}       - Get patient profile
POST /chat                - Agent query endpoint
GET  /tools               - List available tools
GET  /stats               - System metrics
```

---

## References

- OpenAI Embeddings: text-embedding-3-large
- OpenAI GPT-4o for reasoning
- ChromaDB Documentation
- FastAPI Security Patterns
- OWASP LLM Security Guidelines
