import sqlite3


class DatabaseConnection:
    def __init__(self, database_name):
        self.database_name = database_name
    
    def execute_and_commit(self, query, params=None):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        conn.commit()
        conn.close()
    
    def execute_and_fetch(self, query, params=None):
        conn = sqlite3.connect(self.database_name)
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        result = cursor.fetchall()
        conn.close()
        return result


class StudentDatabaseProcessor:
    
    TABLE_CREATION_SQL = """
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        gender TEXT,
        grade INTEGER
    )
    """
    
    INSERT_SQL = "INSERT INTO students (name, age, gender, grade) VALUES (?, ?, ?, ?)"
    SELECT_BY_NAME_SQL = "SELECT * FROM students WHERE name = ?"
    DELETE_BY_NAME_SQL = "DELETE FROM students WHERE name = ?"

    def __init__(self, database_name):
        self.db_conn = DatabaseConnection(database_name)

    def create_student_table(self):
        self.db_conn.execute_and_commit(self.TABLE_CREATION_SQL)

    def insert_student(self, student_data):
        params = (student_data['name'], student_data['age'], 
                 student_data['gender'], student_data['grade'])
        self.db_conn.execute_and_commit(self.INSERT_SQL, params)

    def search_student_by_name(self, name):
        return self.db_conn.execute_and_fetch(self.SELECT_BY_NAME_SQL, (name,))

    def delete_student_by_name(self, name):
        self.db_conn.execute_and_commit(self.DELETE_BY_NAME_SQL, (name,))
