import sqlite3

class BookStorage:
    def __init__(self, db_file):
        self.connection = sqlite3.connect(db_file)
        self.cursor = self.connection.cursor()
        self.initialize_db()

    def initialize_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS bookshelf (
                book_id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                is_borrowed INTEGER
            )
        ''')
        self.connection.commit()

    def insert_book(self, title, author):
        self.cursor.execute('''
            INSERT INTO bookshelf (title, author, is_borrowed)
            VALUES (?, ?, 0)
        ''', (title, author))
        self.connection.commit()

    def delete_book(self, book_id):
        self.cursor.execute('''
            DELETE FROM bookshelf WHERE book_id = ?
        ''', (book_id,))
        self.connection.commit()

    def check_out_book(self, book_id):
        self.cursor.execute('''
            UPDATE bookshelf SET is_borrowed = 1 WHERE book_id = ?
        ''', (book_id,))
        self.connection.commit()

    def check_in_book(self, book_id):
        self.cursor.execute('''
            UPDATE bookshelf SET is_borrowed = 0 WHERE book_id = ?
        ''', (book_id,))
        self.connection.commit()

    def retrieve_all_books(self):
        self.cursor.execute('''
            SELECT * FROM bookshelf
        ''')
        books = self.cursor.fetchall()
        return books
