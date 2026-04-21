import sqlite3


class MovieTicketDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY,
                    movie_name TEXT,
                    theater_name TEXT,
                    seat_number TEXT,
                    customer_name TEXT
                )
            ''')
            self.connection.commit()
        except sqlite3.Error as e:
            self.connection.rollback()
            raise e

    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        try:
            self.cursor.execute('''
                INSERT INTO tickets (movie_name, theater_name, seat_number, customer_name)
                VALUES (?, ?, ?, ?)
            ''', (movie_name, theater_name, seat_number, customer_name))
            self.connection.commit()
        except sqlite3.Error as e:
            self.connection.rollback()
            raise e

    def search_tickets_by_customer(self, customer_name):
        try:
            self.cursor.execute('''
                SELECT * FROM tickets WHERE customer_name = ?
            ''', (customer_name,))
            tickets = self.cursor.fetchall()
            return tickets
        except sqlite3.Error as e:
            raise e

    def delete_ticket(self, ticket_id):
        try:
            self.cursor.execute('''
                DELETE FROM tickets WHERE id = ?
            ''', (ticket_id,))
            self.connection.commit()
        except sqlite3.Error as e:
            self.connection.rollback()
            raise e
