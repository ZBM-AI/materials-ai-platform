"""用户模型 — SQLite."""

import sqlite3
import os
from ..core.security import hash_password


class UserDB:
    def __init__(self, db_path: str = "data/users.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                hashed_password TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def create_user(self, username: str, password: str, email: str = None,
                    role: str = "user") -> bool:
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO users (username, email, hashed_password, role) VALUES (?,?,?,?)",
                (username, email, hash_password(password), role),
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, username: str) -> dict:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, username, email, hashed_password, role FROM users WHERE username=?",
            (username,),
        ).fetchone()
        conn.close()
        if row:
            return {
                "id": row[0], "username": row[1], "email": row[2],
                "hashed_password": row[3], "role": row[4],
            }
        return None


user_db = UserDB()
