import sqlite3
from functools import wraps


def db_operation(commit=True):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            conn = sqlite3.connect(self.database_name)
            cursor = conn.cursor()
            try:
                result = func(self, cursor, *args, **kwargs)
                if commit:
                    conn.commit()
                return result
            finally:
                conn.close()
        return wrapper
    return decorator


class StudentDatabaseProcessor:

    def __init__(self, database_name):
        self.database_name = database_name

    @db_operation(commit=True)
    def create_student_table(self, cursor):
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

    @db_operation(commit=True)
    def insert_student(self, cursor, student_data):
        insert_query = """
        INSERT INTO students (name, age, gender, grade)
        VALUES (?, ?, ?, ?)
        """
        cursor.execute(insert_query,
                       (student_data['name'], student_data['age'], student_data['gender'], student_data['grade']))

    @db_operation(commit=False)
    def search_student_by_name(self, cursor, name):
        select_query = "SELECT * FROM students WHERE name = ?"
        cursor.execute(select_query, (name,))
        return cursor.fetchall()

    @db_operation(commit=True)
    def delete_student_by_name(self, cursor, name):
        delete_query = "DELETE FROM students WHERE name = ?"
        cursor.execute(delete_query, (name,))
