import sqlite3
from enum import IntEnum

class BookStatus(IntEnum):
    BORROWED = 0
    AVAILABLE = 1

class BookManagementDB:
    SCHEMA = '''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY,
            title TEXT,
            author TEXT,
            available INTEGER
        )
    '''
    
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute(self.SCHEMA)
        self.connection.commit()

    def add_book(self, title, author):
        self._execute_and_commit(
            'INSERT INTO books (title, author, available) VALUES (?, ?, ?)',
            (title, author, BookStatus.AVAILABLE)
        )

    def remove_book(self, book_id):
        self._execute_and_commit(
            'DELETE FROM books WHERE id = ?',
            (book_id,)
        )

    def borrow_book(self, book_id):
        self._update_book_status(book_id, BookStatus.BORROWED)

    def return_book(self, book_id):
        self._update_book_status(book_id, BookStatus.AVAILABLE)

    def _update_book_status(self, book_id, status):
        self._execute_and_commit(
            'UPDATE books SET available = ? WHERE id = ?',
            (status, book_id)
        )

    def _execute_and_commit(self, query, params):
        self.cursor.execute(query, params)
        self.connection.commit()

    def search_books(self):
        self.cursor.execute('SELECT * FROM books')
        return self.cursor.fetchall()
