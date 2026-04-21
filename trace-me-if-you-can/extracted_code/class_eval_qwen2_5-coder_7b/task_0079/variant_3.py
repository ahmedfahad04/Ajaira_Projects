class QueryEngine:
    def __init__(self, table_name):
        self.table_name = table_name

    def formulate_query(self, operation, fields=None, data=None, condition=None):
        if operation == "SELECT":
            fields = "* if fields is None else ', '.join(fields)"
            query = f"SELECT {fields} FROM {self.table_name}"
            if condition:
                query += f" WHERE {condition}"
        elif operation == "INSERT":
            fields = ", ".join(data.keys())
            values = ", ".join([f"'{value}'" for value in data.values()])
            query = f"INSERT INTO {self.table_name} ({fields}) VALUES ({values})"
        elif operation == "UPDATE":
            set_clause = ", ".join([f"{field} = '{value}'" for field, value in data.items()])
            query = f"UPDATE {self.table_name} SET {set_clause} WHERE {condition}"
        elif operation == "DELETE":
            query = f"DELETE FROM {self.table_name} WHERE {condition}"
        return query + ";"

    def extract_females_below_age(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.formulate_query("SELECT", condition=condition)

    def extract_by_age_group(self, min_age, max_age):
        condition = f"age BETWEEN {min_age} AND {max_age}"
        return self.formulate_query("SELECT", condition=condition)
