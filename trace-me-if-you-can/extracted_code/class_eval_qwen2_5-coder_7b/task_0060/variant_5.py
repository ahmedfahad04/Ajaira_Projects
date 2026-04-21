import sqlite3

   class FilmSeatDB:
       def __init__(self, db_file):
           self.db_file = db_file
           self.connection = sqlite3.connect(self.db_file)
           self.cursor = self.connection.cursor()
           self.create_database()

       def create_database(self):
           self.cursor.execute('''
               CREATE TABLE IF NOT EXISTS film_seats (
                   id INTEGER PRIMARY KEY,
                   film_name TEXT,
                   theater_name TEXT,
                   seat_number TEXT,
                   customer_name TEXT
               )
           ''')
           self.connection.commit()

       def record_seat_reservation(self, film_name, theater_name, seat_number, customer_name):
           self.cursor.execute('''
               INSERT INTO film_seats (film_name, theater_name, seat_number, customer_name)
               VALUES (?, ?, ?, ?)
           ''', (film_name, theater_name, seat_number, customer_name))
           self.connection.commit()

       def fetch_seats_for_customer(self, customer_name):
           self.cursor.execute('''
               SELECT * FROM film_seats WHERE customer_name = ?
           ''', (customer_name,))
           customer_seats = self.cursor.fetchall()
           return customer_seats

       def cancel_seat_reservation(self, seat_id):
           self.cursor.execute('''
               DELETE FROM film_seats WHERE id = ?
           ''', (seat_id,))
           self.connection.commit()
