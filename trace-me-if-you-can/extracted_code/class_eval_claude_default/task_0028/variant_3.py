import sqlite3
import pandas as pd


class DatabaseProcessor:

    def __init__(self, database_name):
        self.database_name = database_name
        self._connection = None

    def _get_cursor(self):
        if not self._connection:
            self._connection = sqlite3.connect(self.database_name)
        return self._connection.cursor()

    def _commit_and_close(self):
        if self._connection:
            self._connection.commit()
            self._connection.close()
            self._connection = None

    def create_table(self, table_name, key1, key2):
        cursor = self._get_cursor()
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)"
        cursor.execute(create_table_query)
        self._commit_and_close()

    def insert_into_database(self, table_name, data):
        cursor = self._get_cursor()
        for item in data:
            insert_query = f"INSERT INTO {table_name} (name, age) VALUES (?, ?)"
            cursor.execute(insert_query, (item['name'], item['age']))
        self._commit_and_close()

    def search_database(self, table_name, name):
        cursor = self._get_cursor()
        select_query = f"SELECT * FROM {table_name} WHERE name = ?"
        cursor.execute(select_query, (name,))
        result = cursor.fetchall()
        self._commit_and_close()
        
        if result:
            return result
        else:
            return None

    def delete_from_database(self, table_name, name):
        cursor = self._get_cursor()
        delete_query = f"DELETE FROM {table_name} WHERE name = ?"
        cursor.execute(delete_query, (name,))
        self._commit_and_close()
