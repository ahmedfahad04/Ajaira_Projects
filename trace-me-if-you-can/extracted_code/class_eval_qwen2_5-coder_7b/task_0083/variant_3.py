import sqlite3

class StudentHandler:

    def __init__(self, db_location):
        self.db_location = db_location

    def construct_students_table(self):
        db_connection = sqlite3.connect(self.db_location)
        cursor = db_connection.cursor()

        create_command = """
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY,
            student_name TEXT,
            student_age INTEGER,
            student_gender TEXT,
            student_grade INTEGER
        )
        """
        cursor.execute(create_command)

        db_connection.commit()
        db_connection.close()

    def insert_student_entry(self, student_info):
        db_connection = sqlite3.connect(self.db_location)
        cursor = db_connection.cursor()

        insert_command = """
        INSERT INTO students (student_name, student_age, student_gender, student_grade)
        VALUES (?, ?, ?, ?)
        """
        cursor.execute(insert_command,
                       (student_info['name'], student_info['age'], student_info['gender'], student_info['grade']))

        db_connection.commit()
        db_connection.close()

    def retrieve_student_by_name(self, student_name):
        db_connection = sqlite3.connect(self.db_location)
        cursor = db_connection.cursor()

        select_command = "SELECT * FROM students WHERE student_name = ?"
        cursor.execute(select_command, (student_name,))
        student_entries = cursor.fetchall()

        db_connection.close()

        return student_entries

    def remove_student_by_name(self, student_name):
        db_connection = sqlite3.connect(self.db_location)
        cursor = db_connection.cursor()

        delete_command = "DELETE FROM students WHERE student_name = ?"
        cursor.execute(delete_command, (student_name,))

        db_connection.commit()
        db_connection.close()
