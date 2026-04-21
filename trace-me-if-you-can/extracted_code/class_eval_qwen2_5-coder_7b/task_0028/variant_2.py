import sqlite3
import pandas as pd

class DataHandler:
    def __init__(self, db_file):
        self.db_file = db_file

    def open_db_connection(self):
        return sqlite3.connect(self.db_file)

    def generate_table(self, table_name, key1, key2):
        conn = self.open_db_connection()
        cursor = conn.cursor()

        create_table_command = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)"
        cursor.execute(create_table_command)

        conn.commit()
        conn.close()

    def add_entry(self, table_name, data):
        conn = self.open_db_connection()
        cursor = conn.cursor()

        insert_command = f"INSERT INTO {table_name} (name, age) VALUES (?, ?)"
        cursor.executemany(insert_command, data)

        conn.commit()
        conn.close()

    def fetch_entry(self, table_name, name):
        conn = self.open_db_connection()
        cursor = conn.cursor()

        select_command = f"SELECT * FROM {table_name} WHERE name = ?"
        cursor.execute(select_command, (name,))
        fetched_data = cursor.fetchone()

        return fetched_data

    def delete_entry(self, table_name, name):
        conn = self.open_db_connection()
        cursor = conn.cursor()

        delete_command = f"DELETE FROM {table_name} WHERE name = ?"
        cursor.execute(delete_command, (name,))

        conn.commit()
        conn.close()
