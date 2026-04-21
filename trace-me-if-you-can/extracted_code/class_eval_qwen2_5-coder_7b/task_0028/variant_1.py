import sqlite3
import pandas as pd

class DbManager:
    def __init__(self, db_path):
        self.db_path = db_path

    def establish_connection(self):
        return sqlite3.connect(self.db_path)

    def create_schema(self, table_name, key1, key2):
        conn = self.establish_connection()
        cursor = conn.cursor()

        schema_query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)"
        cursor.execute(schema_query)

        conn.commit()
        conn.close()

    def add_record(self, table_name, record):
        conn = self.establish_connection()
        cursor = conn.cursor()

        insert_query = f"INSERT INTO {table_name} (name, age) VALUES (?, ?)"
        cursor.execute(insert_query, (record['name'], record['age']))

        conn.commit()
        conn.close()

    def lookup_record(self, table_name, name):
        conn = self.establish_connection()
        cursor = conn.cursor()

        search_query = f"SELECT * FROM {table_name} WHERE name = ?"
        cursor.execute(search_query, (name,))
        results = cursor.fetchall()

        return results if results else None

    def remove_record(self, table_name, name):
        conn = self.establish_connection()
        cursor = conn.cursor()

        delete_query = f"DELETE FROM {table_name} WHERE name = ?"
        cursor.execute(delete_query, (name,))

        conn.commit()
        conn.close()
