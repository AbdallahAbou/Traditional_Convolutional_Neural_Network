"""
EHR RAG Agent Demo - FastAPI Backend
Synthetic data only - not for clinical use
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import time
import uuid
import os

from database import Database
from vector_store import VectorStore
from agent import EHRAgent
from tools import ToolRegistry

# Lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting EHR RAG Demo Backend...")
    app.state.db = Database()
    app.state.vector_store = VectorStore()
    app.state.tool_registry = ToolRegistry(app.state.db, app.state.vector_store)
    app.state.agent = EHRAgent(app.state.tool_registry)
    print("Backend ready")
    yield
    # Shutdown
    print("Shutting down...")
    app.state.db.close()

app = FastAPI(
    title="EHR RAG Agent Demo",
    description="AI assistant for synthetic EHR data with RAG and tool-use capabilities. **Demo only - not for clinical use.**",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Models ==============

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str

class ChatRequest(BaseModel):
    patient_id: Optional[str] = Field(default=None, description="Patient ID (e.g., P-1001). If None, query is global across all patients.")
    message: str = Field(..., description="User's question")
    history: List[ChatMessage] = Field(default=[], description="Previous messages")
    top_k: int = Field(default=10, ge=1, le=20, description="Number of chunks to retrieve")
    use_reranker: bool = Field(default=True, description="Enable cross-encoder reranking")
    session_id: Optional[str] = Field(default=None, description="Session ID for audit trail")

class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: str
    section: Optional[str]
    text: str
    score: float

class ToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    result_summary: str
    latency_ms: int

class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[ChunkResult]
    tool_calls: List[ToolCall]
    metrics: Dict[str, Any]
    session_id: str

class PatientSummary(BaseModel):
    patient_id: str
    age: int
    sex: str
    primary_diagnosis: str
    disease_stage: Optional[str]

class PatientProfile(BaseModel):
    patient_id: str
    demographics: Dict[str, Any]
    diagnoses: Dict[str, Any]
    vitals: Dict[str, Any]
    labs: Dict[str, Any]
    medications: List[str]
    allergies: List[str]
    contraindications: List[str]
    imaging: Dict[str, Any]
    procedures: List[str]
    dates: Dict[str, Any]


# ============== Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ehr-rag-demo"}


@app.get("/patients", response_model=List[PatientSummary])
async def list_patients(
    limit: int = Query(default=100, le=200),
    offset: int = Query(default=0, ge=0)
):
    """List all patients (summary view)"""
    patients = app.state.db.get_patient_list(limit=limit, offset=offset)
    return patients


@app.get("/patients/{patient_id}", response_model=PatientProfile)
async def get_patient(patient_id: str):
    """Get detailed patient profile"""
    profile = app.state.db.get_patient_profile(patient_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return profile


@app.get("/patients/{patient_id}/documents")
async def get_patient_documents(patient_id: str):
    """Get list of documents for a patient"""
    docs = app.state.db.get_patient_documents(patient_id)
    return {"patient_id": patient_id, "documents": docs}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - Agent answers questions using tools + RAG
    Supports both:
    - Patient-specific queries (patient_id provided)
    - Global queries across all patients (patient_id = None)
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())[:8]
    
    # Validate patient exists if patient_id provided
    if request.patient_id:
        patient = app.state.db.get_patient_profile(request.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not found")
    
    # Run agent
    result = await app.state.agent.run(
        patient_id=request.patient_id,  # Can be None for global queries
        query=request.message,
        history=request.history,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        session_id=session_id
    )
    
    total_time = int((time.time() - start_time) * 1000)
    
    return ChatResponse(
        answer=result["answer"],
        retrieved_chunks=result["chunks"],
        tool_calls=result["tool_calls"],
        metrics={
            "total_latency_ms": total_time,
            "retrieval_latency_ms": result["metrics"].get("retrieval_latency_ms", 0),
            "llm_latency_ms": result["metrics"].get("llm_latency_ms", 0),
            "tokens_in": result["metrics"].get("tokens_in", 0),
            "tokens_out": result["metrics"].get("tokens_out", 0),
            "chunks_retrieved": len(result["chunks"]),
            "tools_called": len(result["tool_calls"])
        },
        session_id=session_id
    )


@app.get("/tools")
async def list_tools():
    """List available agent tools with descriptions"""
    return app.state.tool_registry.get_tool_descriptions()


@app.get("/stats")
async def get_stats():
    """Get database and index statistics"""
    patient_count = app.state.db.get_patient_count()
    doc_count = app.state.db.get_document_count()
    chunk_count = app.state.vector_store.get_chunk_count()
    
    return {
        "patients": patient_count,
        "documents": doc_count,
        "chunks_indexed": chunk_count,
        "embedding_model": app.state.vector_store.model_name,
        "disclaimer": "All data is synthetic - demo only"
    }


@app.get("/audit/{session_id}")
async def get_audit_log(session_id: str):
    """Get audit log for a session"""
    logs = app.state.db.get_audit_logs(session_id)
    return {"session_id": session_id, "logs": logs}


# ============== Tool Endpoints (for direct testing) ==============

@app.get("/tools/patient-profile/{patient_id}")
async def tool_patient_profile(patient_id: str):
    """Direct call to patient profile tool"""
    return app.state.tool_registry.call_tool("get_patient_profile", {"patient_id": patient_id})


@app.get("/tools/medications/{patient_id}")
async def tool_medications(patient_id: str):
    """Direct call to medications tool"""
    return app.state.tool_registry.call_tool("get_medications", {"patient_id": patient_id})


@app.get("/tools/labs/{patient_id}")
async def tool_labs(patient_id: str):
    """Direct call to labs tool"""
    return app.state.tool_registry.call_tool("get_latest_labs", {"patient_id": patient_id})


@app.get("/tools/imaging/{patient_id}")
async def tool_imaging(patient_id: str):
    """Direct call to imaging tool"""
    return app.state.tool_registry.call_tool("get_recent_imaging", {"patient_id": patient_id})


@app.post("/tools/retrieve")
async def tool_retrieve(
    patient_id: str,
    query: str,
    k: int = Query(default=5, ge=1, le=20)
):
    """Direct call to retrieval tool"""
    return app.state.tool_registry.call_tool(
        "retrieve_note_chunks", 
        {"patient_id": patient_id, "query": query, "k": k}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
