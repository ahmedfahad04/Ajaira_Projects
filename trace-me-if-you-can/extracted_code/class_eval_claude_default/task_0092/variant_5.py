import sqlite3


class UserLoginDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def insert_user(self, username, password):
        self.cursor.execute('''
            INSERT INTO users (username, password)
            VALUES (?, ?)
        ''', (username, password))
        self.connection.commit()

    def search_user_by_username(self, username):
        self.cursor.execute('''
            SELECT * FROM users WHERE username = ?
        ''', (username,))
        user = self.cursor.fetchone()
        return user

    def delete_user_by_username(self, username):
        self.cursor.execute('''
            DELETE FROM users WHERE username = ?
        ''', (username,))
        self.connection.commit()

    def validate_user_login(self, username, password):
        self.cursor.execute('''
            SELECT COUNT(*) FROM users WHERE username = ? AND password = ?
        ''', (username, password))
        count = self.cursor.fetchone()[0]
        return count > 0
