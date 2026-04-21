import sqlite3

   class TicketManager:
       def __init__(self, db_name):
           self.db_name = db_name
           self.connection = sqlite3.connect(db_name)
           self.cursor = self.connection.cursor()
           self.initialize_database()

       def initialize_database(self):
           self.cursor.execute('''
               CREATE TABLE IF NOT EXISTS show_seats (
                   id INTEGER PRIMARY KEY,
                   film_name TEXT,
                   venue_name TEXT,
                   seat_position TEXT,
                   client_name TEXT
               )
           ''')
           self.connection.commit()

       def record_seat_occupation(self, film_name, venue_name, seat_position, client_name):
           self.cursor.execute('''
               INSERT INTO show_seats (film_name, venue_name, seat_position, client_name)
               VALUES (?, ?, ?, ?)
           ''', (film_name, venue_name, seat_position, client_name))
           self.connection.commit()

       def fetch_tickets_for_client(self, client_name):
           self.cursor.execute('''
               SELECT * FROM show_seats WHERE client_name = ?
           ''', (client_name,))
           client_tickets = self.cursor.fetchall()
           return client_tickets

       def cancel_seat_occupation(self, seat_id):
           self.cursor.execute('''
               DELETE FROM show_seats WHERE id = ?
           ''', (seat_id,))
           self.connection.commit()
