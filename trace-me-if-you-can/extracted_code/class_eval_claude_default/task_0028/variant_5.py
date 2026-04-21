import sqlite3
import pandas as pd
from typing import List, Dict, Optional, Any


class DatabaseProcessor:

    def __init__(self, database_name: str):
        self.database_name = database_name

    def create_table(self, table_name: str, key1: str, key2: str) -> None:
        self._perform_database_operation(
            lambda cursor: cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {key1} TEXT, {key2} INTEGER)"
            )
        )

    def insert_into_database(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        def insert_operation(cursor):
            insert_query = f"INSERT INTO {table_name} (name, age) VALUES (?, ?)"
            for item in data:
                cursor.execute(insert_query, (item['name'], item['age']))
        
        self._perform_database_operation(insert_operation)

    def search_database(self, table_name: str, name: str) -> Optional[List]:
        def search_operation(cursor):
            select_query = f"SELECT * FROM {table_name} WHERE name = ?"
            cursor.execute(select_query, (name,))
            return cursor.fetchall()
        
        result = self._perform_database_operation(search_operation, return_result=True)
        return result if result else None

    def delete_from_database(self, table_name: str, name: str) -> None:
        self._perform_database_operation(
            lambda cursor: cursor.execute(
                f"DELETE FROM {table_name} WHERE name = ?", (name,)
            )
        )

    def _perform_database_operation(self, operation, return_result: bool = False):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        
        result = operation(cursor)
        
        conn.commit()
        conn.close()
        
        if return_result:
            return result
