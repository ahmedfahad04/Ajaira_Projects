import sqlite3

   class TicketRegistry:
       def __init__(self, database_location):
           self.database_location = database_location
           self.connection = sqlite3.connect(self.database_location)
           self.cursor = self.connection.cursor()
           self.setup_structure()

       def setup_structure(self):
           self.cursor.execute('''
               CREATE TABLE IF NOT EXISTS movie_seats (
                   id INTEGER PRIMARY KEY,
                   movie_title TEXT,
                   theater_name TEXT,
                   seat_location TEXT,
                   customer_name TEXT
               )
           ''')
           self.connection.commit()

       def register_ticket(self, movie_title, theater_name, seat_location, customer_name):
           self.cursor.execute('''
               INSERT INTO movie_seats (movie_title, theater_name, seat_location, customer_name)
               VALUES (?, ?, ?, ?)
           ''', (movie_title, theater_name, seat_location, customer_name))
           self.connection.commit()

       def find_tickets_for_customer(self, customer_name):
           self.cursor.execute('''
               SELECT * FROM movie_seats WHERE customer_name = ?
           ''', (customer_name,))
           customer_seats = self.cursor.fetchall()
           return customer_seats

       def invalidate_seat_reservation(self, seat_id):
           self.cursor.execute('''
               DELETE FROM movie_seats WHERE id = ?
           ''', (seat_id,))
           self.connection.commit()
