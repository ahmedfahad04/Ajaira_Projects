class SQLGenerator:
    QUERY_TEMPLATES = {
        'select': "SELECT {fields} FROM {table}",
        'select_where': "SELECT {fields} FROM {table} WHERE {condition}",
        'insert': "INSERT INTO {table} ({fields}) VALUES ({values})",
        'update': "UPDATE {table} SET {set_clause} WHERE {condition}",
        'delete': "DELETE FROM {table} WHERE {condition}"
    }

    def __init__(self, table_name):
        self.table_name = table_name

    def _format_fields(self, fields):
        return "*" if fields is None else ", ".join(fields)

    def _format_values(self, values):
        return ", ".join([f"'{value}'" for value in values])

    def _format_set_clause(self, data):
        return ", ".join([f"{field} = '{value}'" for field, value in data.items()])

    def select(self, fields=None, condition=None):
        formatted_fields = self._format_fields(fields)
        template_key = 'select_where' if condition else 'select'
        
        params = {'fields': formatted_fields, 'table': self.table_name}
        if condition:
            params['condition'] = condition
            
        return self.QUERY_TEMPLATES[template_key].format(**params) + ";"

    def insert(self, data):
        params = {
            'table': self.table_name,
            'fields': ", ".join(data.keys()),
            'values': self._format_values(data.values())
        }
        return self.QUERY_TEMPLATES['insert'].format(**params) + ";"

    def update(self, data, condition):
        params = {
            'table': self.table_name,
            'set_clause': self._format_set_clause(data),
            'condition': condition
        }
        return self.QUERY_TEMPLATES['update'].format(**params) + ";"

    def delete(self, condition):
        params = {'table': self.table_name, 'condition': condition}
        return self.QUERY_TEMPLATES['delete'].format(**params) + ";"

    def select_female_under_age(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.select(condition=condition)

    def select_by_age_range(self, min_age, max_age):
        condition = f"age BETWEEN {min_age} AND {max_age}"
        return self.select(condition=condition)
