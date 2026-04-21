import sqlite3

class BookDatabase:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_collection()

    def create_collection(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS catalog (
                book_id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                available BOOLEAN
            )
        ''')
        self.connection.commit()

    def insert_book(self, title, author):
        self.cursor.execute('''
            INSERT INTO catalog (title, author, available)
            VALUES (?, ?, True)
        ''', (title, author))
        self.connection.commit()

    def delete_book(self, book_id):
        self.cursor.execute('''
            DELETE FROM catalog WHERE book_id = ?
        ''', (book_id,))
        self.connection.commit()

    def borrow_book(self, book_id):
        self.cursor.execute('''
            UPDATE catalog SET available = False WHERE book_id = ?
        ''', (book_id,))
        self.connection.commit()

    def return_book(self, book_id):
        self.cursor.execute('''
            UPDATE catalog SET available = True WHERE book_id = ?
        ''', (book_id,))
        self.connection.commit()

    def fetch_books(self):
        self.cursor.execute('''
            SELECT * FROM catalog
        ''')
        books = self.cursor.fetchall()
        return books
