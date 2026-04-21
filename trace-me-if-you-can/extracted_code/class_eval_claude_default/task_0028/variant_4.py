import sqlite3
import pandas as pd


class DatabaseProcessor:

    def __init__(self, database_name):
        self.database_name = database_name
        self.queries = {
            'create_table': "CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)",
            'insert': "INSERT INTO {table_name} (name, age) VALUES (?, ?)",
            'select': "SELECT * FROM {table_name} WHERE name = ?",
            'delete': "DELETE FROM {table_name} WHERE name = ?"
        }

    def _execute_with_connection(self, operation, *args, **kwargs):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        
        result = operation(cursor, *args, **kwargs)
        
        conn.commit()
        conn.close()
        return result

    def create_table(self, table_name, key1, key2):
        def create_operation(cursor, table_name, key1, key2):
            query = self.queries['create_table'].format(table_name=table_name, key1=key1, key2=key2)
            cursor.execute(query)
        
        self._execute_with_connection(create_operation, table_name, key1, key2)

    def insert_into_database(self, table_name, data):
        def insert_operation(cursor, table_name, data):
            query = self.queries['insert'].format(table_name=table_name)
            for item in data:
                cursor.execute(query, (item['name'], item['age']))
        
        self._execute_with_connection(insert_operation, table_name, data)

    def search_database(self, table_name, name):
        def search_operation(cursor, table_name, name):
            query = self.queries['select'].format(table_name=table_name)
            cursor.execute(query, (name,))
            return cursor.fetchall()
        
        result = self._execute_with_connection(search_operation, table_name, name)
        return result if result else None

    def delete_from_database(self, table_name, name):
        def delete_operation(cursor, table_name, name):
            query = self.queries['delete'].format(table_name=table_name)
            cursor.execute(query, (name,))
        
        self._execute_with_connection(delete_operation, table_name, name)
