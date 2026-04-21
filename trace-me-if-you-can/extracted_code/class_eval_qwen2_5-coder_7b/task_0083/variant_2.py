import sqlite3

class StudentsDB:

    def __init__(self, db_name):
        self.db_name = db_name

    def setup_students_table(self):
        db_connection = sqlite3.connect(self.db_name)
        cursor = db_connection.cursor()

        table_sql = """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            grade INTEGER
        )
        """
        cursor.execute(table_sql)

        db_connection.commit()
        db_connection.close()

    def add_student_record(self, student_data):
        db_connection = sqlite3.connect(self.db_name)
        cursor = db_connection.cursor()

        add_sql = """
        INSERT INTO students (name, age, gender, grade)
        VALUES (?, ?, ?, ?)
        """
        cursor.execute(add_sql,
                       (student_data['name'], student_data['age'], student_data['gender'], student_data['grade']))

        db_connection.commit()
        db_connection.close()

    def fetch_student_by_name(self, name):
        db_connection = sqlite3.connect(self.db_name)
        cursor = db_connection.cursor()

        select_sql = "SELECT * FROM students WHERE name = ?"
        cursor.execute(select_sql, (name,))
        student_records = cursor.fetchall()

        db_connection.close()

        return student_records

    def remove_student_by_name(self, name):
        db_connection = sqlite3.connect(self.db_name)
        cursor = db_connection.cursor()

        delete_sql = "DELETE FROM students WHERE name = ?"
        cursor.execute(delete_sql, (name,))

        db_connection.commit()
        db_connection.close()
