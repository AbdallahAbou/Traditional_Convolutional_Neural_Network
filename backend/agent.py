"""
EHR Agent - LLM planner with tool-use capabilities
Uses OpenAI GPT-4o for reasoning and tool orchestration
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
import httpx

from tools import ToolRegistry

logger = logging.getLogger(__name__)

# System prompt defining agent behavior and constraints
SYSTEM_PROMPT = """You are an AI clinical data assistant operating over synthetic EHR records.

CRITICAL CONSTRAINTS:
- All patient data is SYNTHETIC and generated for demonstration purposes only
- This system is NOT for clinical use and must never inform real medical decisions
- You can ONLY access data through the provided tools
- You CANNOT select or change the patient context
- You MUST cite evidence for every clinical claim
- You MUST NOT provide diagnostic or treatment recommendations
- You MUST acknowledge when evidence is insufficient

AVAILABLE TOOLS:
1. get_patient_profile - Demographics, diagnoses, and vitals
2. get_medications - Current medications with dosages
3. get_latest_labs - Most recent laboratory values
4. get_recent_imaging - Imaging studies and findings
5. retrieve_note_chunks - Semantic search over clinical notes
6. get_safety_info - Allergies and contraindications
7. get_follow_up_plan - Follow-up appointments and care plan

RESPONSE PROTOCOL:
1. Analyze the query to determine required data sources
2. Use tools to gather relevant information
3. Synthesize findings with explicit citations
4. State confidence level and evidence gaps
5. Include safety notes when applicable

CITATION FORMAT:
- Reference sources as [doc_type:doc_id] e.g., [discharge:P-1001_discharge]
- State when data comes from structured fields vs. clinical notes
- Acknowledge if retrieved evidence is incomplete

PROHIBITED ACTIONS:
- Making diagnostic conclusions
- Recommending treatment changes
- Suggesting medication dosages
- Making prognostic statements
- Providing clinical decision support"""


class EHRAgent:
    """LLM Agent with tool-use for EHR queries"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.model = "gpt-4o"
        logger.info(f"Initialized EHR Agent with {self.model}")
    
    async def run(
        self,
        patient_id: Optional[str],
        query: str,
        history: List[Dict[str, str]] = None,
        top_k: int = 10,
        use_reranker: bool = True,
        session_id: str = ""
    ) -> Dict[str, Any]:
        """
        Execute agent query with tool orchestration.
        
        Args:
            patient_id: Patient identifier (None for global queries)
            query: User query
            history: Conversation history
            top_k: Number of chunks to retrieve
            use_reranker: Whether to apply cross-encoder reranking
            session_id: Audit session identifier
        
        Returns:
            answer: Generated response
            chunks: Retrieved evidence
            tool_calls: Tool execution trace
            metrics: Latency and token counts
        """
        history = history or []
        tool_calls = []
        all_chunks = []
        metrics = {
            "retrieval_latency_ms": 0,
            "llm_latency_ms": 0,
            "rerank_latency_ms": 0,
            "tokens_in": 0,
            "tokens_out": 0
        }
        
        is_global = patient_id is None
        
        # Stage 1: Retrieve relevant chunks
        retrieval_start = time.time()
        retrieval_result = self.tool_registry.call_tool(
            "retrieve_note_chunks",
            {
                "patient_id": patient_id,  # None for global search
                "query": query, 
                "k": top_k,
                "use_reranker": use_reranker
            }
        )
        metrics["retrieval_latency_ms"] = int((time.time() - retrieval_start) * 1000)
        
        if "result" in retrieval_result:
            all_chunks = retrieval_result["result"].get("chunks", [])
            tool_calls.append({
                "tool_name": "retrieve_note_chunks",
                "parameters": {"patient_id": patient_id, "query": query, "k": top_k},
                "result_summary": f"Retrieved {len(all_chunks)} chunks" + (" (global)" if is_global else ""),
                "latency_ms": retrieval_result.get("latency_ms", 0)
            })
        
        # Stage 2: Get structured patient profile (only for patient-specific queries)
        patient_context = ""
        if patient_id:
            profile_result = self.tool_registry.call_tool(
                "get_patient_profile",
                {"patient_id": patient_id}
            )
            if "result" in profile_result and "error" not in profile_result["result"]:
                profile = profile_result["result"]
                patient_context = self._format_profile_context(profile)
                tool_calls.append({
                    "tool_name": "get_patient_profile",
                    "parameters": {"patient_id": patient_id},
                    "result_summary": f"Retrieved profile for {patient_id}",
                    "latency_ms": profile_result.get("latency_ms", 0)
                })
        else:
            patient_context = "GLOBAL QUERY MODE: Searching across all 100 patients in the database."
        
        # Stage 3: Format context for LLM
        chunks_context = self._format_chunks_context(all_chunks)
        
        # Stage 4: Generate response
        llm_start = time.time()
        answer, tokens = await self._call_llm(
            query, patient_context, chunks_context, history
        )
        metrics["llm_latency_ms"] = int((time.time() - llm_start) * 1000)
        metrics["tokens_in"] = tokens.get("input", 0)
        metrics["tokens_out"] = tokens.get("output", 0)
        
        return {
            "answer": answer,
            "chunks": all_chunks,
            "tool_calls": tool_calls,
            "metrics": metrics
        }
    
    def _format_profile_context(self, profile: Dict[str, Any]) -> str:
        """Format patient profile for LLM context"""
        demo = profile.get("demographics", {})
        diag = profile.get("diagnoses", {})
        vitals = profile.get("vitals", {})
        labs = profile.get("labs", {})
        
        lines = [
            f"PATIENT PROFILE: {profile.get('patient_id')}",
            f"Demographics: {demo.get('age')}yo {demo.get('sex')}, BMI {demo.get('bmi')}",
            f"Primary Diagnosis: {diag.get('primary')} (Stage: {diag.get('stage', 'N/A')})",
            f"Secondary Conditions: {', '.join(diag.get('secondary', [])) or 'None'}",
            f"Vitals: BP {vitals.get('bp_systolic')}/{vitals.get('bp_diastolic')}, HR {vitals.get('heart_rate')}, Temp {vitals.get('temperature_c')}C",
            f"Key Labs: ALT {labs.get('alt_u_l')}, AST {labs.get('ast_u_l')}, Bili {labs.get('bilirubin_mg_dl')}, Cr {labs.get('creatinine_mg_dl')}, INR {labs.get('inr')}, Plt {labs.get('platelets_k_ul')}",
            f"Medications: {', '.join(profile.get('medications', [])) or 'None'}",
            f"Allergies: {', '.join(profile.get('allergies', [])) or 'None'}",
            f"Contraindications: {', '.join(profile.get('contraindications', [])) or 'None'}",
            f"Recent Imaging: {profile.get('imaging', {}).get('type')} - {profile.get('imaging', {}).get('findings')}"
        ]
        return "\n".join(lines)
    
    def _format_chunks_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for LLM context with citations"""
        if not chunks:
            return "No relevant clinical notes found."
        
        lines = ["RETRIEVED CLINICAL NOTES:"]
        for i, chunk in enumerate(chunks, 1):
            source = f"[{chunk.get('doc_type', 'note')}:{chunk.get('doc_id', 'unknown')}]"
            section = f"Section: {chunk.get('section')}" if chunk.get('section') else ""
            score = chunk.get('relevance_score', chunk.get('score', 0))
            lines.append(f"\n--- Chunk {i} {source} (relevance: {score:.3f}) {section}")
            lines.append(chunk.get('text', ''))
        
        return "\n".join(lines)
    
    async def _call_llm(
        self, 
        query: str, 
        patient_context: str, 
        chunks_context: str, 
        history: List
    ) -> tuple[str, Dict[str, int]]:
        """Call OpenAI GPT-4o API"""
        
        user_message = f"""Based on the following patient information and clinical notes, please answer this question:

QUESTION: {query}

{patient_context}

{chunks_context}

Provide a thorough answer citing the relevant sources. Use [doc_type:doc_id] format for citations."""
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2048
                }
            )
            
            if response.status_code != 200:
                error_msg = f"OpenAI API error: {response.status_code}"
                logger.error(f"{error_msg} - {response.text}")
                return error_msg, {"input": 0, "output": 0}
            
            data = response.json()
            
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens = {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0)
            }
            
            return answer, tokens
