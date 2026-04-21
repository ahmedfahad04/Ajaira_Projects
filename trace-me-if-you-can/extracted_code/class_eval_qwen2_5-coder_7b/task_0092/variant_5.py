import sqlite3


class UserStore:
    def __init__(self, db_location):
        self.db_location = db_location
        self.setup_store()

    def setup_store(self):
        self.connection = sqlite3.connect(self.db_location)
        self.cursor = self.connection.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        self.connection.commit()

    def add_user(self, username, password):
        self.cursor.execute('''
            INSERT INTO users (username, password)
            VALUES (?, ?)
        ''', (username, password))
        self.connection.commit()

    def find_user(self, username):
        self.cursor.execute('''
            SELECT * FROM users WHERE username = ?
        ''', (username,))
        return self.cursor.fetchone()

    def remove_user(self, username):
        self.cursor.execute('''
            DELETE FROM users WHERE username = ?
        ''', (username,))
        self.connection.commit()

    def validate_login(self, username, password):
        user = self.find_user(username)
        if user and user[1] == password:
            return True
        return False
