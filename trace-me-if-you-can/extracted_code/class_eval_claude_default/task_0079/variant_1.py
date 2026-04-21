class SQLGenerator:
    def __init__(self, table_name):
        self.table_name = table_name
        self._query_builders = {
            'select': self._build_select,
            'insert': self._build_insert,
            'update': self._build_update,
            'delete': self._build_delete
        }

    def _build_select(self, fields=None, condition=None):
        field_str = "*" if fields is None else ", ".join(fields)
        base_sql = f"SELECT {field_str} FROM {self.table_name}"
        return base_sql + (f" WHERE {condition}" if condition else "")

    def _build_insert(self, data):
        fields = ", ".join(data.keys())
        values = ", ".join([f"'{value}'" for value in data.values()])
        return f"INSERT INTO {self.table_name} ({fields}) VALUES ({values})"

    def _build_update(self, data, condition):
        set_clause = ", ".join([f"{field} = '{value}'" for field, value in data.items()])
        return f"UPDATE {self.table_name} SET {set_clause} WHERE {condition}"

    def _build_delete(self, condition):
        return f"DELETE FROM {self.table_name} WHERE {condition}"

    def select(self, fields=None, condition=None):
        return self._query_builders['select'](fields, condition) + ";"

    def insert(self, data):
        return self._query_builders['insert'](data) + ";"

    def update(self, data, condition):
        return self._query_builders['update'](data, condition) + ";"

    def delete(self, condition):
        return self._query_builders['delete'](condition) + ";"

    def select_female_under_age(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.select(condition=condition)

    def select_by_age_range(self, min_age, max_age):
        condition = f"age BETWEEN {min_age} AND {max_age}"
        return self.select(condition=condition)
