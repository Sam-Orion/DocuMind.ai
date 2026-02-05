import sqlite3
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = "documind.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT,
                        status TEXT,
                        upload_timestamp TEXT,
                        processed_timestamp TEXT,
                        result_json TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def save_document(self, doc_id: str, filename: str, status: str = "processing"):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO documents (id, filename, status, upload_timestamp, result_json) VALUES (?, ?, ?, ?, ?)",
                (doc_id, filename, status, timestamp, "{}")
            )
            conn.commit()

    def update_result(self, doc_id: str, result: Dict[str, Any], status: str = "completed"):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            json_str = json.dumps(result)
            cursor.execute(
                "UPDATE documents SET result_json = ?, status = ?, processed_timestamp = ? WHERE id = ?",
                (json_str, status, timestamp, doc_id)
            )
            conn.commit()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                if data['result_json']:
                    data['result_json'] = json.loads(data['result_json'])
                return data
            return None
