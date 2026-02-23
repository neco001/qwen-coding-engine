import os
import duckdb
from datetime import datetime
from pathlib import Path
from platformdirs import user_cache_dir
import logging
import time

logger = logging.getLogger(__name__)

class BillingTracker:
    def __init__(self):
        from platformdirs import user_cache_dir
        self.cache_dir = Path(user_cache_dir("qwen-coding", "Qwen"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "usage_analytics.duckdb"
        self._init_db()

    def _init_db(self):
        try:
            with duckdb.connect(str(self.db_path)) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS usage_stats (
                        timestamp TIMESTAMP,
                        project_name VARCHAR,
                        model_name VARCHAR,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        total_tokens INTEGER
                    )
                """)
        except Exception as e:
            logger.error(f"BillingTracker: Failed to initialize DB: {e}")

    def log_usage(self, project_name: str, model_name: str, prompt_tokens: int, completion_tokens: int):
        total_tokens = prompt_tokens + completion_tokens
        now = datetime.now()
        try:
            # Retry loop for handling DuckDB file locks under concurrent writes
            for attempt in range(5):
                try:
                    with duckdb.connect(str(self.db_path)) as con:
                        con.execute(
                            "INSERT INTO usage_stats VALUES (?, ?, ?, ?, ?, ?)",
                            (now, project_name, model_name, prompt_tokens, completion_tokens, total_tokens)
                        )
                    break
                except duckdb.IOException as e:
                    if "locked" in str(e).lower() and attempt < 4:
                        time.sleep(0.1 * (attempt + 1))
                    else:
                        raise e
        except Exception as e:
            logger.error(f"BillingTracker: Failed to log usage: {e}")

    def get_daily_project_report(self):
        try:
            with duckdb.connect(str(self.db_path)) as con:
                result = con.execute("""
                    SELECT 
                        CAST(timestamp AS DATE) as usage_date,
                        project_name,
                        model_name,
                        SUM(prompt_tokens) as total_prompt,
                        SUM(completion_tokens) as total_completion,
                        SUM(total_tokens) as total_all
                    FROM usage_stats
                    GROUP BY usage_date, project_name, model_name
                    ORDER BY usage_date DESC, project_name, model_name
                """).fetchall()
                
                columns = ["date", "project_name", "model_name", "prompt_tokens", "completion_tokens", "total_tokens"]
                # duckdb dates are returned as datetime.date objects, convert to string
                return [{
                    "date": str(row[0]),
                    "project_name": row[1],
                    "model_name": row[2],
                    "prompt_tokens": row[3],
                    "completion_tokens": row[4],
                    "total_tokens": row[5]
                } for row in result]
        except Exception as e:
            logger.error(f"BillingTracker: Failed to generate report: {e}")
            return []

# Singleton instance
billing_tracker = BillingTracker()
