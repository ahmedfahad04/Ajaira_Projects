class SQLCreator:
    def __init__(self, table):
        self.table = table

    def create_sql_query(self, operation, fields=None, data=None, condition=None):
        if operation == "SELECT":
            fields = "* if fields is None else ', '.join(fields)"
            sql_query = f"SELECT {fields} FROM {self.table}"
            if condition:
                sql_query += f" WHERE {condition}"
        elif operation == "INSERT":
            fields = ", ".join(data.keys())
            values = ", ".join([f"'{value}'" for value in data.values()])
            sql_query = f"INSERT INTO {self.table} ({fields}) VALUES ({values})"
        elif operation == "UPDATE":
            set_clause = ", ".join([f"{field} = '{value}'" for field, value in data.items()])
            sql_query = f"UPDATE {self.table} SET {set_clause} WHERE {condition}"
        elif operation == "DELETE":
            sql_query = f"DELETE FROM {self.table} WHERE {condition}"
        return sql_query + ";"

    def create_female_under_age_query(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.create_sql_query("SELECT", condition=condition)

    def create_age_range_query(self, min_age, max_age):
        condition = f"age BETWEEN {min_age} AND {max_age}"
        return self.create_sql_query("SELECT", condition=condition)
