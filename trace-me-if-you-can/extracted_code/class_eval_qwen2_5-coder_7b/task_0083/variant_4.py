import sqlite3

class StudentRegistry:

    def __init__(self, db_url):
        self.db_url = db_url

    def create_students_table(self):
        connection = sqlite3.connect(self.db_url)
        cursor = connection.cursor()

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY,
            student_name TEXT,
            student_age INTEGER,
            student_gender TEXT,
            student_grade INTEGER
        )
        """
        cursor.execute(create_table_sql)

        connection.commit()
        connection.close()

    def add_student(self, student_details):
        connection = sqlite3.connect(self.db_url)
        cursor = connection.cursor()

        insert_sql = """
        INSERT INTO students (student_name, student_age, student_gender, student_grade)
        VALUES (?, ?, ?, ?)
        """
        cursor.execute(insert_sql,
                       (student_details['name'], student_details['age'], student_details['gender'], student_details['grade']))

        connection.commit()
        connection.close()

    def search_student_by_name(self, student_name):
        connection = sqlite3.connect(self.db_url)
        cursor = connection.cursor()

        select_sql = "SELECT * FROM students WHERE student_name = ?"
        cursor.execute(select_sql, (student_name,))
        search_results = cursor.fetchall()

        connection.close()

        return search_results

    def delete_student_by_name(self, student_name):
        connection = sqlite3.connect(self.db_url)
        cursor = connection.cursor()

        delete_sql = "DELETE FROM students WHERE student_name = ?"
        cursor.execute(delete_sql, (student_name,))

        connection.commit()
        connection.close()
