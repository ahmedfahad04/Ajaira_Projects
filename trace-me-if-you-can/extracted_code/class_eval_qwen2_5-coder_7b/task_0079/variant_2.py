class QueryComposer:
    def __init__(self, table):
        self.table = table

    def generate(self, query_type, fields=None, data=None, condition=None):
        if query_type == "SELECT":
            fields = "* if fields is None else ', '.join(fields)"
            query = f"SELECT {fields} FROM {self.table}"
            if condition:
                query += f" WHERE {condition}"
        elif query_type == "INSERT":
            fields = ", ".join(data.keys())
            values = ", ".join([f"'{value}'" for value in data.values()])
            query = f"INSERT INTO {self.table} ({fields}) VALUES ({values})"
        elif query_type == "UPDATE":
            set_clause = ", ".join([f"{field} = '{value}'" for field, value in data.items()])
            query = f"UPDATE {self.table} SET {set_clause} WHERE {condition}"
        elif query_type == "DELETE":
            query = f"DELETE FROM {self.table} WHERE {condition}"
        return query + ";"

    def query_females_below_age(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.generate("SELECT", condition=condition)

    def query_by_age_range(self, min_age, max_age):
        condition = f"age BETWEEN {min_age} AND {max_age}"
        return self.generate("SELECT", condition=condition)
