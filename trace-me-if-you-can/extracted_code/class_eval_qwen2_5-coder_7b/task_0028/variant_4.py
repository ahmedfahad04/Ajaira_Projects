import sqlite3
import pandas as pd

class SQLiteManager:
    def __init__(self, db_file):
        self.db_file = db_file

    def get_connection(self):
        return sqlite3.connect(self.db_file)

    def create_table(self, table_name, key1, key2):
        conn = self.get_connection()
        cursor = conn.cursor()

        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)"
        cursor.execute(create_table_query)

        conn.commit()
        conn.close()

    def insert_data(self, table_name, data):
        conn = self.get_connection()
        cursor = conn.cursor()

        insert_query = f"INSERT INTO {table_name} (name, age) VALUES (?, ?)"
        for item in data:
            cursor.execute(insert_query, (item['name'], item['age']))

        conn.commit()
        conn.close()

    def search_data(self, table_name, name):
        conn = self.get_connection()
        cursor = conn.cursor()

        select_query = f"SELECT * FROM {table_name} WHERE name = ?"
        cursor.execute(select_query, (name,))
        result = cursor.fetchall()

        return result if result else None

    def delete_data(self, table_name, name):
        conn = self.get_connection()
        cursor = conn.cursor()

        delete_query = f"DELETE FROM {table_name} WHERE name = ?"
        cursor.execute(delete_query, (name,))

        conn.commit()
        conn.close()
