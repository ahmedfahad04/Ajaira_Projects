class SQLGenerator:
    def __init__(self, table_name):
        self.table_name = table_name

    def _execute_query_generation(self, query_type, **kwargs):
        query_map = {
            'SELECT': lambda: self._generate_select(**kwargs),
            'INSERT': lambda: self._generate_insert(**kwargs),
            'UPDATE': lambda: self._generate_update(**kwargs),
            'DELETE': lambda: self._generate_delete(**kwargs)
        }
        return query_map[query_type]() + ";"

    def _generate_select(self, fields=None, condition=None):
        parts = ["SELECT"]
        parts.append("*" if fields is None else ", ".join(fields))
        parts.extend(["FROM", self.table_name])
        
        if condition is not None:
            parts.extend(["WHERE", condition])
        
        return " ".join(parts)

    def _generate_insert(self, data):
        field_list = ", ".join(data.keys())
        value_list = ", ".join([f"'{value}'" for value in data.values()])
        return f"INSERT INTO {self.table_name} ({field_list}) VALUES ({value_list})"

    def _generate_update(self, data, condition):
        assignments = ", ".join([f"{field} = '{value}'" for field, value in data.items()])
        return f"UPDATE {self.table_name} SET {assignments} WHERE {condition}"

    def _generate_delete(self, condition):
        return f"DELETE FROM {self.table_name} WHERE {condition}"

    def select(self, fields=None, condition=None):
        return self._execute_query_generation('SELECT', fields=fields, condition=condition)

    def insert(self, data):
        return self._execute_query_generation('INSERT', data=data)

    def update(self, data, condition):
        return self._execute_query_generation('UPDATE', data=data, condition=condition)

    def delete(self, condition):
        return self._execute_query_generation('DELETE', condition=condition)

    def select_female_under_age(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.select(condition=condition)

    def select_by_age_range(self, min_age, max_age):
        condition = f"age BETWEEN {min_age} AND {max_age}"
        return self.select(condition=condition)
