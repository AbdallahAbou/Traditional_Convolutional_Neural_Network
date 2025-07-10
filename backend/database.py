"""
Database connection and queries for EHR Demo
Read-only access pattern for safety
"""

import mysql.connector
from mysql.connector import pooling
import os
from typing import List, Dict, Any, Optional
import json


class Database:
    def __init__(self):
        self.pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="ehr_pool",
            pool_size=5,
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "ehr_agent"),
            password=os.getenv("MYSQL_PASSWORD", "ehrAgent2024!"),
            database=os.getenv("MYSQL_DATABASE", "ehr_demo"),
            charset='utf8mb4',
            collation='utf8mb4_unicode_ci'
        )
    
    def _get_connection(self):
        return self.pool.get_connection()
    
    def close(self):
        pass  # Pool handles cleanup
    
    # ============== Patient Queries ==============
    
    def get_patient_list(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get list of patients (summary)"""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT patient_id, age, sex, primary_diagnosis, disease_stage
                FROM patients
                ORDER BY patient_id
                LIMIT %s OFFSET %s
            """, (limit, offset))
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
    
    def get_patient_profile(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get full patient profile (structured)"""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            # Parse medications, allergies, contraindications
            meds = self._parse_list(row.get('current_medications', ''))
            allergies = self._parse_list(row.get('allergies', ''))
            contraindications = self._parse_list(row.get('contraindications', ''))
            procedures = self._parse_list(row.get('procedures', ''))
            secondary = self._parse_list(row.get('secondary_conditions', ''))
            
            return {
                "patient_id": row['patient_id'],
                "demographics": {
                    "age": row['age'],
                    "sex": row['sex'],
                    "height_cm": float(row['height_cm']) if row.get('height_cm') else None,
                    "weight_kg": float(row['weight_kg']) if row.get('weight_kg') else None,
                    "bmi": float(row['bmi']) if row.get('bmi') else None,
                    "smoking_status": row.get('smoking_status')
                },
                "diagnoses": {
                    "primary": row['primary_diagnosis'],
                    "stage": row.get('disease_stage'),
                    "secondary": secondary,
                    "icd10_codes": row.get('icd10_codes', '').split('; ') if row.get('icd10_codes') else []
                },
                "vitals": {
                    "bp_systolic": row.get('bp_systolic'),
                    "bp_diastolic": row.get('bp_diastolic'),
                    "heart_rate": row.get('heart_rate'),
                    "temperature_c": float(row['temperature_c']) if row.get('temperature_c') else None
                },
                "labs": {
                    "alt_u_l": row.get('alt_u_l'),
                    "ast_u_l": row.get('ast_u_l'),
                    "bilirubin_mg_dl": float(row['bilirubin_mg_dl']) if row.get('bilirubin_mg_dl') else None,
                    "afp_ng_ml": float(row['afp_ng_ml']) if row.get('afp_ng_ml') else None,
                    "creatinine_mg_dl": float(row['creatinine_mg_dl']) if row.get('creatinine_mg_dl') else None,
                    "egfr_ml_min": float(row['egfr_ml_min']) if row.get('egfr_ml_min') else None,
                    "inr": float(row['inr']) if row.get('inr') else None,
                    "platelets_k_ul": row.get('platelets_k_ul')
                },
                "medications": meds,
                "allergies": allergies,
                "contraindications": contraindications,
                "imaging": {
                    "type": row.get('recent_imaging_type'),
                    "date": str(row['recent_imaging_date']) if row.get('recent_imaging_date') else None,
                    "findings": row.get('key_imaging_findings')
                },
                "procedures": procedures,
                "dates": {
                    "admission": str(row['admission_date']) if row.get('admission_date') else None,
                    "discharge": str(row['discharge_date']) if row.get('discharge_date') else None,
                    "follow_up": str(row['follow_up_date']) if row.get('follow_up_date') else None,
                    "follow_up_plan": row.get('follow_up_plan')
                }
            }
        finally:
            cursor.close()
            conn.close()
    
    def get_patient_medications(self, patient_id: str) -> List[Dict[str, str]]:
        """Get medications for a patient"""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT current_medications, allergies, contraindications
                FROM patients WHERE patient_id = %s
            """, (patient_id,))
            row = cursor.fetchone()
            if not row:
                return []
            
            meds = self._parse_list(row.get('current_medications', ''))
            return [{"medication": m, "status": "active"} for m in meds]
        finally:
            cursor.close()
            conn.close()
    
    def get_patient_labs(self, patient_id: str) -> Dict[str, Any]:
        """Get lab values for a patient"""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT alt_u_l, ast_u_l, bilirubin_mg_dl, afp_ng_ml,
                       creatinine_mg_dl, egfr_ml_min, inr, platelets_k_ul
                FROM patients WHERE patient_id = %s
            """, (patient_id,))
            row = cursor.fetchone()
            if not row:
                return {}
            
            # Convert Decimal types
            return {
                "ALT": {"value": row.get('alt_u_l'), "unit": "U/L", "reference": "7-56"},
                "AST": {"value": row.get('ast_u_l'), "unit": "U/L", "reference": "10-40"},
                "Bilirubin": {"value": float(row['bilirubin_mg_dl']) if row.get('bilirubin_mg_dl') else None, "unit": "mg/dL", "reference": "0.1-1.2"},
                "AFP": {"value": float(row['afp_ng_ml']) if row.get('afp_ng_ml') else None, "unit": "ng/mL", "reference": "<10"},
                "Creatinine": {"value": float(row['creatinine_mg_dl']) if row.get('creatinine_mg_dl') else None, "unit": "mg/dL", "reference": "0.7-1.3"},
                "eGFR": {"value": float(row['egfr_ml_min']) if row.get('egfr_ml_min') else None, "unit": "mL/min/1.73m²", "reference": ">90"},
                "INR": {"value": float(row['inr']) if row.get('inr') else None, "unit": "", "reference": "0.8-1.2"},
                "Platelets": {"value": row.get('platelets_k_ul'), "unit": "K/µL", "reference": "150-400"}
            }
        finally:
            cursor.close()
            conn.close()
    
    def get_patient_imaging(self, patient_id: str) -> Dict[str, Any]:
        """Get imaging info for a patient"""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT recent_imaging_type, recent_imaging_date, key_imaging_findings
                FROM patients WHERE patient_id = %s
            """, (patient_id,))
            row = cursor.fetchone()
            if not row:
                return {}
            return {
                "type": row.get('recent_imaging_type'),
                "date": str(row['recent_imaging_date']) if row.get('recent_imaging_date') else None,
                "findings": row.get('key_imaging_findings')
            }
        finally:
            cursor.close()
            conn.close()
    
    # ============== Document Queries ==============
    
    def get_patient_documents(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get documents for a patient"""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT doc_id, doc_type, title, doc_date, author, word_count
                FROM documents WHERE patient_id = %s
                ORDER BY doc_date DESC
            """, (patient_id,))
            rows = cursor.fetchall()
            for row in rows:
                if row.get('doc_date'):
                    row['doc_date'] = str(row['doc_date'])
            return rows
        finally:
            cursor.close()
            conn.close()
    
    def get_document_count(self) -> int:
        """Get total document count"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]
        finally:
            cursor.close()
            conn.close()
    
    def get_patient_count(self) -> int:
        """Get total patient count"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM patients")
            return cursor.fetchone()[0]
        finally:
            cursor.close()
            conn.close()
    
    # ============== Audit Logging ==============
    
    def log_tool_call(self, session_id: str, tool_name: str, patient_id: str,
                      parameters: Dict, result_summary: str, latency_ms: int,
                      tokens_in: int = 0, tokens_out: int = 0):
        """Log a tool call for audit/observability"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO audit_log (session_id, tool_name, patient_id, parameters, 
                                       result_summary, latency_ms, tokens_in, tokens_out)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (session_id, tool_name, patient_id, json.dumps(parameters),
                  result_summary[:500], latency_ms, tokens_in, tokens_out))
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def get_audit_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Get audit logs for a session"""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT tool_name, patient_id, parameters, result_summary, 
                       latency_ms, tokens_in, tokens_out, created_at
                FROM audit_log WHERE session_id = %s
                ORDER BY created_at
            """, (session_id,))
            rows = cursor.fetchall()
            for row in rows:
                if row.get('parameters'):
                    row['parameters'] = json.loads(row['parameters'])
                if row.get('created_at'):
                    row['created_at'] = str(row['created_at'])
            return rows
        finally:
            cursor.close()
            conn.close()
    
    # ============== Helpers ==============
    
    def _parse_list(self, text: str) -> List[str]:
        """Parse pipe or semicolon separated list"""
        if not text or text.lower() == 'none':
            return []
        # Try pipe first, then semicolon
        if '|' in text:
            items = [x.strip() for x in text.split('|')]
        elif ';' in text:
            items = [x.strip() for x in text.split(';')]
        else:
            items = [text.strip()]
        return [x for x in items if x and x.lower() != 'none']
