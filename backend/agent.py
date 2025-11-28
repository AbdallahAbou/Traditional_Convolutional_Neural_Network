"""
EHR Agent - LLM planner with tool-use capabilities
Uses OpenAI GPT-4o with function calling for dynamic tool orchestration
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
5. retrieve_note_chunks - Semantic search over clinical notes (RAG)
6. get_safety_info - Allergies and contraindications
7. get_follow_up_plan - Follow-up appointments and care plan
8. run_sql_query - Execute SQL queries for aggregations, counts, filtering (read-only SELECT only)

WHEN TO USE SQL QUERIES (run_sql_query):
- Counting patients by criteria (e.g., "how many patients have diabetes")
- Aggregating statistics (e.g., "average age of patients with cirrhosis")
- Finding patients matching specific criteria
- Complex filtering that other tools cannot handle
- Global queries across all patients
- Use table: patients (columns: patient_id, age, sex, primary_diagnosis, disease_stage, current_medications, allergies, etc.)
- Use table: documents (columns: doc_id, patient_id, doc_type, content)

WHEN TO USE RAG (retrieve_note_chunks):
- Searching for specific information in clinical notes
- Finding mentions of symptoms, procedures, or treatments
- Looking up qualitative information from narratives

RESPONSE PROTOCOL:
1. Analyze the query to determine required data sources
2. Use tools to gather relevant information (call multiple tools if needed)
3. Synthesize findings with explicit citations
4. State confidence level and evidence gaps
5. Include safety notes when applicable

CITATION FORMAT:
- Reference sources as [doc_type:doc_id] e.g., [discharge:P-1001_discharge]
- State when data comes from structured fields vs. clinical notes
- For SQL results, cite as [SQL query result]

PROHIBITED ACTIONS:
- Making diagnostic conclusions
- Recommending treatment changes
- Suggesting medication dosages
- Making prognostic statements
- Providing clinical decision support"""


class EHRAgent:
    """LLM Agent with dynamic tool calling for EHR queries"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.model = "gpt-4o"
        self.max_tool_iterations = 5  # Prevent infinite loops
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
        Execute agent query with dynamic tool orchestration.
        
        The LLM decides which tools to call via OpenAI function calling.
        Supports multiple tool calls in a single response.
        """
        history = history or []
        tool_calls_trace = []
        all_chunks = []
        metrics = {
            "retrieval_latency_ms": 0,
            "llm_latency_ms": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "tool_iterations": 0
        }
        
        is_global = patient_id is None
        
        # Build context message
        context_msg = self._build_context_message(patient_id, query)
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_msg}
        ]
        
        # Get tools in OpenAI format
        tools = self.tool_registry.get_openai_tools()
        
        # Agentic loop - let LLM call tools until it has enough info
        llm_start = time.time()
        total_tokens_in = 0
        total_tokens_out = 0
        assistant_message = {"content": ""}
        
        for iteration in range(self.max_tool_iterations):
            metrics["tool_iterations"] = iteration + 1
            
            # Call LLM
            response, tokens = await self._call_llm_with_tools(messages, tools)
            total_tokens_in += tokens.get("input", 0)
            total_tokens_out += tokens.get("output", 0)
            
            if "error" in response:
                return {
                    "answer": f"Error: {response['error']}",
                    "chunks": [],
                    "tool_calls": tool_calls_trace,
                    "metrics": metrics
                }
            
            assistant_message = response["message"]
            messages.append(assistant_message)
            
            # Check if LLM wants to call tools
            if not assistant_message.get("tool_calls"):
                # LLM is done, extract final answer
                break
            
            # Process each tool call
            for tool_call in assistant_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                # Inject defaults for retrieve_note_chunks
                if tool_name == "retrieve_note_chunks":
                    if "k" not in tool_args:
                        tool_args["k"] = top_k
                    if "use_reranker" not in tool_args:
                        tool_args["use_reranker"] = use_reranker
                    # If patient_id not specified, use context patient_id (or None for global)
                    if "patient_id" not in tool_args:
                        tool_args["patient_id"] = patient_id
                
                # Execute tool
                tool_start = time.time()
                result = self.tool_registry.call_tool(tool_name, tool_args)
                tool_latency = int((time.time() - tool_start) * 1000)
                
                # Track metrics
                if tool_name == "retrieve_note_chunks":
                    metrics["retrieval_latency_ms"] += tool_latency
                    if "result" in result:
                        chunks = result["result"].get("chunks", [])
                        all_chunks.extend(chunks)
                
                # Build result summary
                result_summary = self._summarize_tool_result(tool_name, result)
                
                tool_calls_trace.append({
                    "tool_name": tool_name,
                    "parameters": tool_args,
                    "result_summary": result_summary,
                    "latency_ms": tool_latency
                })
                
                # Add tool result to messages
                tool_result_content = json.dumps(result.get("result", result), default=str)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result_content[:8000]  # Limit context size
                })
        
        metrics["llm_latency_ms"] = int((time.time() - llm_start) * 1000)
        metrics["tokens_in"] = total_tokens_in
        metrics["tokens_out"] = total_tokens_out
        
        # Extract final answer
        final_answer = assistant_message.get("content", "")
        if not final_answer:
            final_answer = "I was unable to generate a response. Please try rephrasing your question."
        
        # Deduplicate chunks
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            chunk_id = chunk.get("doc_id", "") + chunk.get("text", "")[:50]
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
        
        return {
            "answer": final_answer,
            "chunks": unique_chunks[:top_k],  # Limit returned chunks
            "tool_calls": tool_calls_trace,
            "metrics": metrics
        }
    
    def _build_context_message(self, patient_id: Optional[str], query: str) -> str:
        """Build the initial context message for the LLM"""
        if patient_id:
            return f"""You are answering a question about patient {patient_id}.

QUESTION: {query}

Use the available tools to gather the information needed to answer this question. 
Start by retrieving relevant data using the appropriate tools."""
        else:
            return f"""You are answering a GLOBAL question across ALL patients in the database (100 synthetic patients).

QUESTION: {query}

Use the available tools to gather the information needed to answer this question.
For counting or aggregation queries, use run_sql_query with appropriate SQL.
For searching clinical notes, use retrieve_note_chunks without patient_id for global search."""
    
    def _summarize_tool_result(self, tool_name: str, result: Dict) -> str:
        """Create a brief summary of tool result for UI display"""
        if "error" in result:
            return f"Error: {result['error']}"
        
        r = result.get("result", {})
        
        if tool_name == "retrieve_note_chunks":
            count = r.get("count", 0)
            scope = r.get("patient_id", "GLOBAL")
            return f"Retrieved {count} chunks ({scope})"
        
        elif tool_name == "run_sql_query":
            if "error" in r:
                return f"SQL Error: {r['error']}"
            row_count = r.get("row_count", 0)
            return f"SQL returned {row_count} rows"
        
        elif tool_name == "get_patient_profile":
            pid = r.get("patient_id", "unknown")
            diag = r.get("diagnoses", {}).get("primary", "")[:30]
            return f"Profile: {pid} - {diag}"
        
        elif tool_name in ["get_medications", "get_safety_info"]:
            meds = len(r.get("current_medications", []))
            allergies = len(r.get("allergies", []))
            return f"{meds} medications, {allergies} allergies"
        
        elif tool_name == "get_latest_labs":
            return f"Labs for {r.get('patient_id', 'unknown')}"
        
        elif tool_name == "get_recent_imaging":
            imaging = r.get("imaging", {})
            return f"Imaging: {imaging.get('type', 'none')}"
        
        elif tool_name == "get_follow_up_plan":
            return f"Follow-up: {r.get('follow_up_date', 'N/A')}"
        
        return "Completed"
    
    async def _call_llm_with_tools(
        self, 
        messages: List[Dict], 
        tools: List[Dict]
    ) -> tuple[Dict, Dict[str, int]]:
        """Call OpenAI API with function calling enabled"""
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "temperature": 0.3,
                        "max_tokens": 2048
                    }
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {"error": error_msg}, {"input": 0, "output": 0}
                
                data = response.json()
                
                message = data["choices"][0]["message"]
                usage = data.get("usage", {})
                tokens = {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0)
                }
                
                return {"message": message}, tokens
                
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return {"error": str(e)}, {"input": 0, "output": 0}
