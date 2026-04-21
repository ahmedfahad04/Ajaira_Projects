import sqlite3

class BookManagementDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        table_schema = [
            "CREATE TABLE IF NOT EXISTS books (",
            "    id INTEGER PRIMARY KEY,",
            "    title TEXT,",
            "    author TEXT,",
            "    available INTEGER",
            ")"
        ]
        self.cursor.execute(" ".join(table_schema))
        self.connection.commit()

    def add_book(self, title, author):
        book_data = {'title': title, 'author': author, 'available': 1}
        self.cursor.execute(
            "INSERT INTO books (title, author, available) VALUES (:title, :author, :available)",
            book_data
        )
        self.connection.commit()

    def remove_book(self, book_id):
        self.cursor.execute("DELETE FROM books WHERE id = :id", {'id': book_id})
        self.connection.commit()

    def borrow_book(self, book_id):
        self._set_availability(book_id, False)

    def return_book(self, book_id):
        self._set_availability(book_id, True)

    def _set_availability(self, book_id, is_available):
        availability_value = 1 if is_available else 0
        self.cursor.execute(
            "UPDATE books SET available = :available WHERE id = :id",
            {'available': availability_value, 'id': book_id}
        )
        self.connection.commit()

    def search_books(self):
        self.cursor.execute("SELECT * FROM books")
        return [tuple(row) for row in self.cursor.fetchall()]
