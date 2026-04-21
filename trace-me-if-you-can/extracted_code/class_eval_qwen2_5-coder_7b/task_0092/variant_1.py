import sqlite3


class AccountManager:
    def __init__(self, db_name):
        self.db_name = db_name
        self.setup_db()

    def setup_db(self):
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        self.connection.commit()

    def register_user(self, user_name, user_pass):
        self.cursor.execute('''
            INSERT INTO users (username, password)
            VALUES (?, ?)
        ''', (user_name, user_pass))
        self.connection.commit()

    def lookup_user(self, user_name):
        self.cursor.execute('''
            SELECT * FROM users WHERE username = ?
        ''', (user_name,))
        return self.cursor.fetchone()

    def remove_user(self, user_name):
        self.cursor.execute('''
            DELETE FROM users WHERE username = ?
        ''', (user_name,))
        self.connection.commit()

    def authenticate_user(self, user_name, user_pass):
        user_info = self.lookup_user(user_name)
        if user_info and user_info[1] == user_pass:
            return True
        return False
