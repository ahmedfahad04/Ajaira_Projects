import sqlite3
from contextlib import contextmanager


class UserLoginDB:
    def __init__(self, db_name):
        self.db_name = db_name

    @contextmanager
    def _get_connection(self):
        connection = sqlite3.connect(self.db_name)
        try:
            yield connection
        finally:
            connection.close()

    def insert_user(self, username, password):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, password)
                VALUES (?, ?)
            ''', (username, password))
            conn.commit()

    def search_user_by_username(self, username):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM users WHERE username = ?
            ''', (username,))
            return cursor.fetchone()

    def delete_user_by_username(self, username):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM users WHERE username = ?
            ''', (username,))
            conn.commit()

    def validate_user_login(self, username, password):
        user = self.search_user_by_username(username)
        if user is not None and user[1] == password:
            return True
        return False
