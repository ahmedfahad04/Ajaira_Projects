import sqlite3
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Ticket:
    id: int
    movie_name: str
    theater_name: str
    seat_number: str
    customer_name: str


class MovieTicketDB:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
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

    def insert_ticket(self, movie_name, theater_name, seat_number, customer_name):
        ticket_data = {
            'movie_name': movie_name,
            'theater_name': theater_name,
            'seat_number': seat_number,
            'customer_name': customer_name
        }
        
        columns = ', '.join(ticket_data.keys())
        placeholders = ', '.join(['?' for _ in ticket_data])
        values = tuple(ticket_data.values())
        
        query = f'INSERT INTO tickets ({columns}) VALUES ({placeholders})'
        self.cursor.execute(query, values)
        self.connection.commit()

    def search_tickets_by_customer(self, customer_name):
        query_params = {'customer_name': customer_name}
        self.cursor.execute('''
            SELECT * FROM tickets WHERE customer_name = ?
        ''', tuple(query_params.values()))
        return self.cursor.fetchall()

    def delete_ticket(self, ticket_id):
        deletion_criteria = {'id': ticket_id}
        self.cursor.execute('''
            DELETE FROM tickets WHERE id = ?
        ''', tuple(deletion_criteria.values()))
        self.connection.commit()
