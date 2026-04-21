import sqlite3


class UserLoginDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def insert_user(self, username, password):
        self._execute_and_commit(
            'INSERT INTO users (username, password) VALUES (?, ?)',
            (username, password)
        )

    def search_user_by_username(self, username):
        self.cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        return self.cursor.fetchone()

    def delete_user_by_username(self, username):
        self._execute_and_commit(
            'DELETE FROM users WHERE username = ?',
            (username,)
        )

    def validate_user_login(self, username, password):
        user = self.search_user_by_username(username)
        return user is not None and user[1] == password

    def _execute_and_commit(self, query, params):
        self.cursor.execute(query, params)
        self.connection.commit()
