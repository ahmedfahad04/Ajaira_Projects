import sqlite3
import pandas as pd

class DatabaseConnector:
    def __init__(self, db_path):
        self.db_path = db_path

    def connect_to_db(self):
        return sqlite3.connect(self.db_path)

    def setup_table(self, table_name, key1, key2):
        conn = self.connect_to_db()
        cursor = conn.cursor()

        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)"
        cursor.execute(create_table_sql)

        conn.commit()
        conn.close()

    def insert_into_table(self, table_name, data):
        conn = self.connect_to_db()
        cursor = conn.cursor()

        insert_sql = f"INSERT INTO {table_name} (name, age) VALUES (?, ?)"
        for record in data:
            cursor.execute(insert_sql, (record['name'], record['age']))

        conn.commit()
        conn.close()

    def retrieve_from_table(self, table_name, name):
        conn = self.connect_to_db()
        cursor = conn.cursor()

        select_sql = f"SELECT * FROM {table_name} WHERE name = ?"
        cursor.execute(select_sql, (name,))
        result = cursor.fetchall()

        return result if result else None

    def delete_from_table(self, table_name, name):
        conn = self.connect_to_db()
        cursor = conn.cursor()

        delete_sql = f"DELETE FROM {table_name} WHERE name = ?"
        cursor.execute(delete_sql, (name,))

        conn.commit()
        conn.close()
