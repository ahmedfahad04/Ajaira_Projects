import sqlite3


class UserDatabase:
    def __init__(self, db_file):
        self.db_file = db_file
        self.initialize_db()

    def initialize_db(self):
        self.connection = sqlite3.connect(self.db_file)
        self.cursor = self.connection.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                user_name TEXT NOT NULL,
                user_password TEXT NOT NULL
            )
        ''')
        self.connection.commit()

    def add_user(self, user_name, user_password):
        self.cursor.execute('''
            INSERT INTO users (user_name, user_password)
            VALUES (?, ?)
        ''', (user_name, user_password))
        self.connection.commit()

    def locate_user(self, user_name):
        self.cursor.execute('''
            SELECT * FROM users WHERE user_name = ?
        ''', (user_name,))
        return self.cursor.fetchone()

    def remove_user(self, user_name):
        self.cursor.execute('''
            DELETE FROM users WHERE user_name = ?
        ''', (user_name,))
        self.connection.commit()

    def check_user_login(self, user_name, user_password):
        user_info = self.locate_user(user_name)
        if user_info and user_info[1] == user_password:
            return True
        return False
