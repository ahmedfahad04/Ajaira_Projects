import sqlite3

class Library:
    def __init__(self, db_location):
        self.connection = sqlite3.connect(db_location)
        self.cursor = self.connection.cursor()
        self.setup_library()

    def setup_library(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection (
                id INTEGER PRIMARY KEY,
                name TEXT,
                creator TEXT,
                is_checked_out INTEGER
            )
        ''')
        self.connection.commit()

    def add_item(self, name, creator):
        self.cursor.execute('''
            INSERT INTO collection (name, creator, is_checked_out)
            VALUES (?, ?, 0)
        ''', (name, creator))
        self.connection.commit()

    def remove_item(self, item_id):
        self.cursor.execute('''
            DELETE FROM collection WHERE id = ?
        ''', (item_id,))
        self.connection.commit()

    def checkout_item(self, item_id):
        self.cursor.execute('''
            UPDATE collection SET is_checked_out = 1 WHERE id = ?
        ''', (item_id,))
        self.connection.commit()

    def checkin_item(self, item_id):
        self.cursor.execute('''
            UPDATE collection SET is_checked_out = 0 WHERE id = ?
        ''', (item_id,))
        self.connection.commit()

    def get_all_items(self):
        self.cursor.execute('''
            SELECT * FROM collection
        ''')
        items = self.cursor.fetchall()
        return items
