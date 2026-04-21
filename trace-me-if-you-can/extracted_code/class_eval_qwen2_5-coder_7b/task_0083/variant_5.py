import sqlite3

class StudentDB:
    def __init__(self, db_name):
        self.db_name = db_name

    def create_student_table(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER,
                gender TEXT,
                grade INTEGER
            )
            """)
            conn.commit()

    def insert_student(self, data):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO students (name, age, gender, grade)
            VALUES (?, ?, ?, ?)
            """, (data['name'], data['age'], data['gender'], data['grade']))
            conn.commit()

    def search_student(self, name):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students WHERE name = ?", (name,))
            return cursor.fetchall()

    def delete_student(self, name):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM students WHERE name = ?", (name,))
            conn.commit()
