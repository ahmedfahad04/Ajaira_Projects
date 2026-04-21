import sqlite3


class UserLoginDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.connection.row_factory = sqlite3.Row

    def insert_user(self, username, password):
        with self.connection:
            self.connection.execute('''
                INSERT INTO users (username, password)
                VALUES (?, ?)
            ''', (username, password))

    def search_user_by_username(self, username):
        cursor = self.connection.execute('''
            SELECT * FROM users WHERE username = ?
        ''', (username,))
        return cursor.fetchone()

    def delete_user_by_username(self, username):
        with self.connection:
            self.connection.execute('''
                DELETE FROM users WHERE username = ?
            ''', (username,))

    def validate_user_login(self, username, password):
        user = self.search_user_by_username(username)
        if user is not None and user['password'] == password:
            return True
        return False
