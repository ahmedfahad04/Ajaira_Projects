import sqlite3
from contextlib import contextmanager


class MovieTicketDB:
    def __init__(self, db_name):
        self.db_name = db_name
        self.create_table()

    @contextmanager
    def get_connection(self):
        connection = sqlite3.connect(self.db_name)
        try:
            yield connection
        finally:
            connection.close()

    def create_table(self):
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY,
                    movie_name TEXT,
                    theater_name TEXT,
                    seat_number TEXT,
                    customer_name TEXT
                )
            ''')
            conn.commit()

    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO tickets (movie_name, theater_name, seat_number, customer_name)
                VALUES (?, ?, ?, ?)
            ''', (movie_name, theater_name, seat_number, customer_name))
            conn.commit()

    def search_tickets_by_customer(self, customer_name):
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM tickets WHERE customer_name = ?
            ''', (customer_name,))
            return cursor.fetchall()

    def delete_ticket(self, ticket_id):
        with self.get_connection() as conn:
            conn.execute('''
                DELETE FROM tickets WHERE id = ?
            ''', (ticket_id,))
            conn.commit()
