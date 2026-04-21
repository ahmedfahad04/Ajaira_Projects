import sqlite3


class UserProfileDB:
    def __init__(self, filename):
        self.filename = filename
        self.connect_and_initialize()

    def connect_and_initialize(self):
        self.connection = sqlite3.connect(self.filename)
        self.cursor = self.connection.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                user_name TEXT NOT NULL,
                user_pass TEXT NOT NULL
            )
        ''')
        self.connection.commit()

    def add_user(self, user_name, user_pass):
        self.cursor.execute('''
            INSERT INTO users (user_name, user_pass)
            VALUES (?, ?)
        ''', (user_name, user_pass))
        self.connection.commit()

    def find_user(self, user_name):
        self.cursor.execute('''
            SELECT * FROM users WHERE user_name = ?
        ''', (user_name,))
        return self.cursor.fetchone()

    def remove_user(self, user_name):
        self.cursor.execute('''
            DELETE FROM users WHERE user_name = ?
        ''', (user_name,))
        self.connection.commit()

    def check_credentials(self, user_name, user_pass):
        user_data = self.find_user(user_name)
        if user_data and user_data[1] == user_pass:
            return True
        return False
