import sqlite3

class LibraryManager:
    def __init__(self, db_path):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.setup_database()

    def setup_database(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS library (
                book_id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                is_available INTEGER
            )
        ''')
        self.connection.commit()

    def add_book(self, title, author):
        self.cursor.execute('''
            INSERT INTO library (title, author, is_available)
            VALUES (?, ?, 1)
        ''', (title, author))
        self.connection.commit()

    def delete_book(self, book_id):
        self.cursor.execute('''
            DELETE FROM library WHERE book_id = ?
        ''', (book_id,))
        self.connection.commit()

    def borrow_book(self, book_id):
        self.cursor.execute('''
            UPDATE library SET is_available = 0 WHERE book_id = ?
        ''', (book_id,))
        self.connection.commit()

    def return_book(self, book_id):
        self.cursor.execute('''
            UPDATE library SET is_available = 1 WHERE book_id = ?
        ''', (book_id,))
        self.connection.commit()

    def list_books(self):
        self.cursor.execute('''
            SELECT * FROM library
        ''')
        books = self.cursor.fetchall()
        return books
