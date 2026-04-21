import sqlite3


class MovieTicketDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.operations = {
            'create': self._create_table_operation,
            'insert': self._insert_ticket_operation,
            'search': self._search_tickets_operation,
            'delete': self._delete_ticket_operation
        }
        self.create_table()

    def create_table(self):
        self.operations['create']()

    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        self.operations['insert'](movie_name, theater_name, seat_number, customer_name)

    def search_tickets_by_customer(self, customer_name):
        return self.operations['search'](customer_name)

    def delete_ticket(self, ticket_id):
        self.operations['delete'](ticket_id)

    def _create_table_operation(self):
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

    def _insert_ticket_operation(self, movie_name, theater_name, seat_number, customer_name):
        self.cursor.execute('''
            INSERT INTO tickets (movie_name, theater_name, seat_number, customer_name)
            VALUES (?, ?, ?, ?)
        ''', (movie_name, theater_name, seat_number, customer_name))
        self.connection.commit()

    def _search_tickets_operation(self, customer_name):
        self.cursor.execute('''
            SELECT * FROM tickets WHERE customer_name = ?
        ''', (customer_name,))
        return self.cursor.fetchall()

    def _delete_ticket_operation(self, ticket_id):
        self.cursor.execute('''
            DELETE FROM tickets WHERE id = ?
        ''', (ticket_id,))
        self.connection.commit()
