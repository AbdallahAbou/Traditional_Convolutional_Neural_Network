"""
Agent Tools Registry
Safe, wrapped functions that the LLM can call
No direct SQL or filesystem access
"""

import time
from typing import Dict, Any, List, Callable
from database import Database
from vector_store import VectorStore


class ToolRegistry:
    """Registry of tools available to the agent"""
    
    def __init__(self, db: Database, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools"""
        
        # Tool: Get Patient Profile
        self.tools["get_patient_profile"] = {
            "name": "get_patient_profile",
            "description": "Get comprehensive patient profile including demographics, diagnoses, vitals, labs, medications, allergies, and imaging summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient ID (e.g., P-1001)"
                    }
                },
                "required": ["patient_id"]
            },
            "function": self._get_patient_profile
        }
        
        # Tool: Get Medications
        self.tools["get_medications"] = {
            "name": "get_medications",
            "description": "Get current medications for a patient, along with allergies and contraindications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient ID"
                    }
                },
                "required": ["patient_id"]
            },
            "function": self._get_medications
        }
        
        # Tool: Get Latest Labs
        self.tools["get_latest_labs"] = {
            "name": "get_latest_labs",
            "description": "Get most recent lab values for a patient including liver function, kidney function, and coagulation tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient ID"
                    }
                },
                "required": ["patient_id"]
            },
            "function": self._get_latest_labs
        }
        
        # Tool: Get Recent Imaging
        self.tools["get_recent_imaging"] = {
            "name": "get_recent_imaging",
            "description": "Get recent imaging studies and findings for a patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient ID"
                    }
                },
                "required": ["patient_id"]
            },
            "function": self._get_recent_imaging
        }
        
        # Tool: Retrieve Note Chunks (RAG)
        self.tools["retrieve_note_chunks"] = {
            "name": "retrieve_note_chunks",
            "description": "Search clinical notes for relevant information using semantic search. Returns most relevant text chunks with source citations. Can search within a specific patient or globally across all patients.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient ID to search within (optional - omit for global search)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query describing the information needed"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 5, max 20)",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            "function": self._retrieve_note_chunks
        }
        
        # Tool: Get All Patient Allergies & Contraindications
        self.tools["get_safety_info"] = {
            "name": "get_safety_info",
            "description": "Get patient allergies and medication contraindications for safety checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient ID"
                    }
                },
                "required": ["patient_id"]
            },
            "function": self._get_safety_info
        }
        
        # Tool: Get Follow-up Plan
        self.tools["get_follow_up_plan"] = {
            "name": "get_follow_up_plan",
            "description": "Get the patient's follow-up appointments and care plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient ID"
                    }
                },
                "required": ["patient_id"]
            },
            "function": self._get_follow_up_plan
        }
    
    # ============== Tool Implementations ==============
    
    def _get_patient_profile(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient profile"""
        profile = self.db.get_patient_profile(patient_id)
        if not profile:
            return {"error": f"Patient {patient_id} not found"}
        return profile
    
    def _get_medications(self, patient_id: str) -> Dict[str, Any]:
        """Get medications with safety info"""
        profile = self.db.get_patient_profile(patient_id)
        if not profile:
            return {"error": f"Patient {patient_id} not found"}
        
        return {
            "patient_id": patient_id,
            "current_medications": profile.get("medications", []),
            "allergies": profile.get("allergies", []),
            "contraindications": profile.get("contraindications", [])
        }
    
    def _get_latest_labs(self, patient_id: str) -> Dict[str, Any]:
        """Get latest lab values"""
        labs = self.db.get_patient_labs(patient_id)
        if not labs:
            return {"error": f"Patient {patient_id} not found or no labs"}
        return {"patient_id": patient_id, "labs": labs}
    
    def _get_recent_imaging(self, patient_id: str) -> Dict[str, Any]:
        """Get recent imaging"""
        imaging = self.db.get_patient_imaging(patient_id)
        if not imaging:
            return {"error": f"Patient {patient_id} not found or no imaging"}
        return {"patient_id": patient_id, "imaging": imaging}
    
    def _retrieve_note_chunks(
        self, 
        patient_id: str = None, 
        query: str = "", 
        k: int = 10,
        use_reranker: bool = True
    ) -> Dict[str, Any]:
        """
        Multi-stage retrieval over clinical notes.
        
        Stage 1: Dense retrieval with k=50 candidates
        Stage 2: Cross-encoder reranking to top k results
        
        Args:
            patient_id: Patient scope (None for global search)
            query: Search query
            k: Final number of results (after reranking)
            use_reranker: Whether to apply reranking
        """
        # Enforce limits
        k = min(k, 20)
        
        chunks = self.vector_store.search(
            query=query,
            patient_id=patient_id,  # None triggers global search
            k=50,  # Stage 1: high recall
            rerank=use_reranker,
            rerank_top_n=k  # Stage 2: precision
        )
        
        return {
            "patient_id": patient_id if patient_id else "GLOBAL",
            "query": query,
            "chunks": chunks,
            "count": len(chunks),
            "reranked": use_reranker
        }
    
    def _get_safety_info(self, patient_id: str) -> Dict[str, Any]:
        """Get safety-relevant information"""
        profile = self.db.get_patient_profile(patient_id)
        if not profile:
            return {"error": f"Patient {patient_id} not found"}
        
        return {
            "patient_id": patient_id,
            "allergies": profile.get("allergies", []),
            "contraindications": profile.get("contraindications", []),
            "current_medications": profile.get("medications", [])
        }
    
    def _get_follow_up_plan(self, patient_id: str) -> Dict[str, Any]:
        """Get follow-up plan"""
        profile = self.db.get_patient_profile(patient_id)
        if not profile:
            return {"error": f"Patient {patient_id} not found"}
        
        dates = profile.get("dates", {})
        return {
            "patient_id": patient_id,
            "admission_date": dates.get("admission"),
            "discharge_date": dates.get("discharge"),
            "follow_up_date": dates.get("follow_up"),
            "follow_up_plan": dates.get("follow_up_plan"),
            "primary_diagnosis": profile.get("diagnoses", {}).get("primary"),
            "procedures": profile.get("procedures", [])
        }
    
    # ============== Registry Methods ==============
    
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name with parameters"""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        start = time.time()
        try:
            result = self.tools[tool_name]["function"](**parameters)
            latency = int((time.time() - start) * 1000)
            return {"result": result, "latency_ms": latency}
        except Exception as e:
            return {"error": str(e)}
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Get tool descriptions for LLM"""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
            for t in self.tools.values()
        ]
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get tools in OpenAI function calling format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"]
                }
            }
            for t in self.tools.values()
        ]
