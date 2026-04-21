import sqlite3

class BookStore:
    def __init__(self, db_path):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.initialize_store()

    def initialize_store(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory (
                item_id INTEGER PRIMARY KEY,
                title TEXT,
                creator TEXT,
                is_reserved BOOLEAN
            )
        ''')
        self.connection.commit()

    def add_item(self, title, creator):
        self.cursor.execute('''
            INSERT INTO inventory (title, creator, is_reserved)
            VALUES (?, ?, False)
        ''', (title, creator))
        self.connection.commit()

    def remove_item(self, item_id):
        self.cursor.execute('''
            DELETE FROM inventory WHERE item_id = ?
        ''', (item_id,))
        self.connection.commit()

    def reserve_item(self, item_id):
        self.cursor.execute('''
            UPDATE inventory SET is_reserved = True WHERE item_id = ?
        ''', (item_id,))
        self.connection.commit()

    def release_item(self, item_id):
        self.cursor.execute('''
            UPDATE inventory SET is_reserved = False WHERE item_id = ?
        ''', (item_id,))
        self.connection.commit()

    def get_all_items(self):
        self.cursor.execute('''
            SELECT * FROM inventory
        ''')
        items = self.cursor.fetchall()
        return items
