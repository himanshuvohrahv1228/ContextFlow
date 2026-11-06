import sqlite3
from pathlib import Path
from datetime import datetime


class ChatHistory:
    """Saves simple question-answer history in a local SQLite database."""

    def __init__(self, database_path: str = "data/chat_history.db"):
        self.database_path = str(database_path)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            connection.commit()

    def save(self, question: str, answer: str):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO conversations (question, answer, created_at) VALUES (?, ?, ?)",
                (question, answer, datetime.now().isoformat(timespec="seconds"))
            )
            connection.commit()

    def get_recent(self, limit: int = 20):
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT question, answer, created_at FROM conversations ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [
            {"question": row[0], "answer": row[1], "created_at": row[2]}
            for row in rows
        ]
