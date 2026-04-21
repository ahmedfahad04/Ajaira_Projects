import sqlite3
from contextlib import contextmanager


class StudentDatabaseProcessor:

    def __init__(self, database_name):
        self.database_name = database_name

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.database_name)
        try:
            yield conn
        finally:
            conn.close()

    def create_student_table(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            create_table_query = """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER,
                gender TEXT,
                grade INTEGER
            )
            """
            cursor.execute(create_table_query)
            conn.commit()

    def insert_student(self, student_data):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            insert_query = """
            INSERT INTO students (name, age, gender, grade)
            VALUES (?, ?, ?, ?)
            """
            cursor.execute(insert_query,
                           (student_data['name'], student_data['age'], student_data['gender'], student_data['grade']))
            conn.commit()

    def search_student_by_name(self, name):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            select_query = "SELECT * FROM students WHERE name = ?"
            cursor.execute(select_query, (name,))
            return cursor.fetchall()

    def delete_student_by_name(self, name):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            delete_query = "DELETE FROM students WHERE name = ?"
            cursor.execute(delete_query, (name,))
            conn.commit()
