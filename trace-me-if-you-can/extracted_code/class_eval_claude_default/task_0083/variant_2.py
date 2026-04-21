import sqlite3


class StudentDatabaseProcessor:

    def __init__(self, database_name):
        self.database_name = database_name
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        conn = sqlite3.connect(self.database_name)
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
        conn.close()

    def create_student_table(self):
        # Table is already created in __init__
        pass

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

    def insert_student(self, student_data):
        insert_query = """
        INSERT INTO students (name, age, gender, grade)
        VALUES (?, ?, ?, ?)
        """
        self._execute_query(insert_query,
                           (student_data['name'], student_data['age'], student_data['gender'], student_data['grade']))

    def search_student_by_name(self, name):
        select_query = "SELECT * FROM students WHERE name = ?"
        return self._execute_query(select_query, (name,), fetch=True)

    def delete_student_by_name(self, name):
        delete_query = "DELETE FROM students WHERE name = ?"
        self._execute_query(delete_query, (name,))
