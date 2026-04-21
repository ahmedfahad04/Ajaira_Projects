import sqlite3

class StudentManager:

    def __init__(self, db_path):
        self.db_path = db_path

    def build_students_table(self):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        table_creation_command = """
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY,
            student_name TEXT,
            student_age INTEGER,
            student_gender TEXT,
            student_grade INTEGER
        )
        """
        cursor.execute(table_creation_command)

        connection.commit()
        connection.close()

    def add_student(self, student_info):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        insert_command = """
        INSERT INTO students (student_name, student_age, student_gender, student_grade)
        VALUES (?, ?, ?, ?)
        """
        cursor.execute(insert_command,
                       (student_info['name'], student_info['age'], student_info['gender'], student_info['grade']))

        connection.commit()
        connection.close()

    def lookup_student_by_name(self, student_name):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        search_command = "SELECT * FROM students WHERE student_name = ?"
        cursor.execute(search_command, (student_name,))
        found_students = cursor.fetchall()

        connection.close()

        return found_students

    def erase_student_by_name(self, student_name):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        delete_command = "DELETE FROM students WHERE student_name = ?"
        cursor.execute(delete_command, (student_name,))

        connection.commit()
        connection.close()
