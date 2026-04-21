import sqlite3
from typing import Optional, Tuple


class UserLoginDB:
    QUERIES = {
        'insert': 'INSERT INTO users (username, password) VALUES (?, ?)',
        'select': 'SELECT * FROM users WHERE username = ?',
        'delete': 'DELETE FROM users WHERE username = ?'
    }

    def __init__(self, db_name: str):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def insert_user(self, username: str, password: str) -> None:
        self.cursor.execute(self.QUERIES['insert'], (username, password))
        self.connection.commit()

    def search_user_by_username(self, username: str) -> Optional[Tuple]:
        self.cursor.execute(self.QUERIES['select'], (username,))
        return self.cursor.fetchone()

    def delete_user_by_username(self, username: str) -> None:
        self.cursor.execute(self.QUERIES['delete'], (username,))
        self.connection.commit()

    def validate_user_login(self, username: str, password: str) -> bool:
        user_data = self.search_user_by_username(username)
        return bool(user_data and user_data[1] == password)
