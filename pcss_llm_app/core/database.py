import sqlite3
import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                created_at TIMESTAMP,
                model TEXT,
                mode TEXT
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def create_conversation(self, title, model, mode="chat"):
        conn = self._get_connection()
        cursor = conn.cursor()
        created_at = datetime.datetime.now()
        cursor.execute(
            'INSERT INTO conversations (title, created_at, model, mode) VALUES (?, ?, ?, ?)',
            (title, created_at, model, mode)
        )
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return conversation_id

    def add_message(self, conversation_id, role, content):
        conn = self._get_connection()
        cursor = conn.cursor()
        timestamp = datetime.datetime.now()
        cursor.execute(
            'INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)',
            (conversation_id, role, content, timestamp)
        )
        conn.commit()
        conn.close()

    def get_conversations(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM conversations ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_messages(self, conversation_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC', (conversation_id,))
        rows = cursor.fetchall()
        conn.close()
        return rows
