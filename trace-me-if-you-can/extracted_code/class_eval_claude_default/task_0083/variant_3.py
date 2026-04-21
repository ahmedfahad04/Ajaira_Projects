import sqlite3
from typing import Dict, List, Tuple, Any


class StudentDatabaseProcessor:
    STUDENT_SCHEMA = {
        'table_name': 'students',
        'columns': ['id', 'name', 'age', 'gender', 'grade'],
        'create_sql': """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER,
                gender TEXT,
                grade INTEGER
            )
        """
    }

    def __init__(self, database_name: str):
        self.database_name = database_name

    def _execute_sql(self, sql: str, params: Tuple = None, fetch_results: bool = False) -> List[Tuple]:
        with sqlite3.connect(self.database_name) as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            if fetch_results:
                return cursor.fetchall()
            return []

    def create_student_table(self) -> None:
        self._execute_sql(self.STUDENT_SCHEMA['create_sql'])

    def insert_student(self, student_data: Dict[str, Any]) -> None:
        insert_sql = f"""
        INSERT INTO {self.STUDENT_SCHEMA['table_name']} (name, age, gender, grade)
        VALUES (?, ?, ?, ?)
        """
        params = (student_data['name'], student_data['age'], 
                 student_data['gender'], student_data['grade'])
        self._execute_sql(insert_sql, params)

    def search_student_by_name(self, name: str) -> List[Tuple]:
        select_sql = f"SELECT * FROM {self.STUDENT_SCHEMA['table_name']} WHERE name = ?"
        return self._execute_sql(select_sql, (name,), fetch_results=True)

    def delete_student_by_name(self, name: str) -> None:
        delete_sql = f"DELETE FROM {self.STUDENT_SCHEMA['table_name']} WHERE name = ?"
        self._execute_sql(delete_sql, (name,))
