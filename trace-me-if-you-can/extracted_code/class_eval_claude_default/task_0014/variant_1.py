import sqlite3
from contextlib import contextmanager

class BookManagementDB:
    def __init__(self, db_name):
        self.db_name = db_name
        self._initialize_database()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_name)
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_database(self):
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    author TEXT,
                    available INTEGER
                )
            ''')
            conn.commit()

    def add_book(self, title, author):
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO books (title, author, available)
                VALUES (?, ?, 1)
            ''', (title, author))
            conn.commit()

    def remove_book(self, book_id):
        with self._get_connection() as conn:
            conn.execute('''
                DELETE FROM books WHERE id = ?
            ''', (book_id,))
            conn.commit()

    def borrow_book(self, book_id):
        with self._get_connection() as conn:
            conn.execute('''
                UPDATE books SET available = 0 WHERE id = ?
            ''', (book_id,))
            conn.commit()

    def return_book(self, book_id):
        with self._get_connection() as conn:
            conn.execute('''
                UPDATE books SET available = 1 WHERE id = ?
            ''', (book_id,))
            conn.commit()

    def search_books(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM books
            ''')
            return cursor.fetchall()
