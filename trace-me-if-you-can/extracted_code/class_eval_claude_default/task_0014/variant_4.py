import sqlite3

class BookManagementDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.queries = {
            'create_table': '''
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    author TEXT,
                    available INTEGER
                )
            ''',
            'add_book': 'INSERT INTO books (title, author, available) VALUES (?, ?, 1)',
            'remove_book': 'DELETE FROM books WHERE id = ?',
            'update_availability': 'UPDATE books SET available = ? WHERE id = ?',
            'search_books': 'SELECT * FROM books'
        }
        self.create_table()

    def create_table(self):
        self._execute_query('create_table')

    def add_book(self, title, author):
        self._execute_query('add_book', (title, author))

    def remove_book(self, book_id):
        self._execute_query('remove_book', (book_id,))

    def borrow_book(self, book_id):
        self._execute_query('update_availability', (0, book_id))

    def return_book(self, book_id):
        self._execute_query('update_availability', (1, book_id))

    def _execute_query(self, query_key, params=None):
        if params:
            self.cursor.execute(self.queries[query_key], params)
        else:
            self.cursor.execute(self.queries[query_key])
        self.connection.commit()

    def search_books(self):
        self.cursor.execute(self.queries['search_books'])
        books = self.cursor.fetchall()
        return books
