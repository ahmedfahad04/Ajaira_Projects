import sqlite3

   class TicketSystem:
       def __init__(self, db_path):
           self.db_path = db_path
           self.connection = sqlite3.connect(self.db_path)
           self.cursor = self.connection.cursor()
           self.setup_database()

       def setup_database(self):
           self.cursor.execute('''
               CREATE TABLE IF NOT EXISTS movie_tickets (
                   id INTEGER PRIMARY KEY,
                   movie_title TEXT,
                   theater TEXT,
                   seat TEXT,
                   buyer_name TEXT
               )
           ''')
           self.connection.commit()

       def record_ticket(self, movie_title, theater, seat, buyer_name):
           self.cursor.execute('''
               INSERT INTO movie_tickets (movie_title, theater, seat, buyer_name)
               VALUES (?, ?, ?, ?)
           ''', (movie_title, theater, seat, buyer_name))
           self.connection.commit()

       def locate_tickets_for_buyer(self, buyer_name):
           self.cursor.execute('''
               SELECT * FROM movie_tickets WHERE buyer_name = ?
           ''', (buyer_name,))
           found_tickets = self.cursor.fetchall()
           return found_tickets

       def invalidate_ticket(self, ticket_id):
           self.cursor.execute('''
               DELETE FROM movie_tickets WHERE id = ?
           ''', (ticket_id,))
           self.connection.commit()
