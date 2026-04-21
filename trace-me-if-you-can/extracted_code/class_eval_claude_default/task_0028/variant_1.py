import sqlite3
import pandas as pd
from contextlib import contextmanager


class DatabaseProcessor:

    def __init__(self, database_name):
        self.database_name = database_name

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.database_name)
        try:
            yield conn
        finally:
            conn.close()

    def create_table(self, table_name, key1, key2):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)"
            cursor.execute(create_table_query)
            conn.commit()

    def insert_into_database(self, table_name, data):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for item in data:
                insert_query = f"INSERT INTO {table_name} (name, age) VALUES (?, ?)"
                cursor.execute(insert_query, (item['name'], item['age']))
            conn.commit()

    def search_database(self, table_name, name):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            select_query = f"SELECT * FROM {table_name} WHERE name = ?"
            cursor.execute(select_query, (name,))
            result = cursor.fetchall()
            return result if result else None

    def delete_from_database(self, table_name, name):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            delete_query = f"DELETE FROM {table_name} WHERE name = ?"
            cursor.execute(delete_query, (name,))
            conn.commit()
