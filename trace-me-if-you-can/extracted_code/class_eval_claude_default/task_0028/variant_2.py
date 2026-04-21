import sqlite3
import pandas as pd


class DatabaseProcessor:

    def __init__(self, database_name):
        self.database_name = database_name

    def _execute_query(self, query, params=None, fetch=False):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        result = None
        if fetch:
            result = cursor.fetchall()
        
        conn.commit()
        conn.close()
        return result

    def create_table(self, table_name, key1, key2):
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)"
        self._execute_query(create_table_query)

    def insert_into_database(self, table_name, data):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        
        insert_query = f"INSERT INTO {table_name} (name, age) VALUES (?, ?)"
        cursor.executemany(insert_query, [(item['name'], item['age']) for item in data])
        
        conn.commit()
        conn.close()

    def search_database(self, table_name, name):
        select_query = f"SELECT * FROM {table_name} WHERE name = ?"
        result = self._execute_query(select_query, (name,), fetch=True)
        return result if result else None

    def delete_from_database(self, table_name, name):
        delete_query = f"DELETE FROM {table_name} WHERE name = ?"
        self._execute_query(delete_query, (name,))
