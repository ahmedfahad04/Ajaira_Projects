import sqlite3


class MovieTicketDB:
    TABLE_SCHEMA = '''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY,
            movie_name TEXT,
            theater_name TEXT,
            seat_number TEXT,
            customer_name TEXT
        )
    '''
    
    INSERT_QUERY = '''
        INSERT INTO tickets (movie_name, theater_name, seat_number, customer_name)
        VALUES (?, ?, ?, ?)
    '''
    
    SEARCH_QUERY = 'SELECT * FROM tickets WHERE customer_name = ?'
    DELETE_QUERY = 'DELETE FROM tickets WHERE id = ?'

    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self._execute_and_commit(self.TABLE_SCHEMA)

    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        self._execute_and_commit(self.INSERT_QUERY, 
                                (movie_name, theater_name, seat_number, customer_name))

    def search_tickets_by_customer(self, customer_name):
        self.cursor.execute(self.SEARCH_QUERY, (customer_name,))
        return self.cursor.fetchall()

    def delete_ticket(self, ticket_id):
        self._execute_and_commit(self.DELETE_QUERY, (ticket_id,))

    def _execute_and_commit(self, query, params=None):
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        self.connection.commit()
