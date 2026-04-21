import sqlite3

   class CinemaTicketDB:
       def __init__(self, database_url):
           self.database_url = database_url
           self.connection = sqlite3.connect(self.database_url)
           self.cursor = self.connection.cursor()
           self.build_schema()

       def build_schema(self):
           self.cursor.execute('''
               CREATE TABLE IF NOT EXISTS cinema_tickets (
                   id INTEGER PRIMARY KEY,
                   film_title TEXT,
                   venue_name TEXT,
                   seat_number TEXT,
                   purchaser_name TEXT
               )
           ''')
           self.connection.commit()

       def log_ticket_purchase(self, film_title, venue_name, seat_number, purchaser_name):
           self.cursor.execute('''
               INSERT INTO cinema_tickets (film_title, venue_name, seat_number, purchaser_name)
               VALUES (?, ?, ?, ?)
           ''', (film_title, venue_name, seat_number, purchaser_name))
           self.connection.commit()

       def retrieve_tickets_for_purchaser(self, purchaser_name):
           self.cursor.execute('''
               SELECT * FROM cinema_tickets WHERE purchaser_name = ?
           ''', (purchaser_name,))
           purchaser_tickets = self.cursor.fetchall()
           return purchaser_tickets

       def remove_ticket_purchase(self, ticket_id):
           self.cursor.execute('''
               DELETE FROM cinema_tickets WHERE id = ?
           ''', (ticket_id,))
           self.connection.commit()
