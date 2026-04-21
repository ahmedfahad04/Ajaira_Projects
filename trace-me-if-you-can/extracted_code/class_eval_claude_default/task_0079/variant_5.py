class SQLGenerator:
    def __init__(self, table_name):
        self.table_name = table_name
        self.query_cache = {}

    def _get_cache_key(self, operation, **params):
        return f"{operation}_{hash(frozenset(params.items()) if params else 0)}"

    def _build_sql_components(self, operation, **kwargs):
        components = {'table': self.table_name, 'terminator': ';'}
        
        if operation == 'select':
            components['action'] = 'SELECT'
            components['fields'] = '*' if kwargs.get('fields') is None else ', '.join(kwargs['fields'])
            components['from_clause'] = f"FROM {self.table_name}"
            components['where_clause'] = f" WHERE {kwargs['condition']}" if kwargs.get('condition') else ""
            
        elif operation == 'insert':
            data = kwargs['data']
            components['action'] = 'INSERT INTO'
            components['fields'] = f"({', '.join(data.keys())})"
            components['values'] = f"VALUES ({', '.join([f\"'{v}'\" for v in data.values()])})"
            
        elif operation == 'update':
            data, condition = kwargs['data'], kwargs['condition']
            components['action'] = 'UPDATE'
            components['set_clause'] = f"SET {', '.join([f\"{k} = '{v}'\" for k, v in data.items()])}"
            components['where_clause'] = f" WHERE {condition}"
            
        elif operation == 'delete':
            components['action'] = 'DELETE'
            components['from_clause'] = f"FROM {self.table_name}"
            components['where_clause'] = f" WHERE {kwargs['condition']}"
            
        return components

    def _assemble_query(self, operation, **kwargs):
        components = self._build_sql_components(operation, **kwargs)
        
        if operation == 'select':
            return f"{components['action']} {components['fields']} {components['from_clause']}{components['where_clause']}{components['terminator']}"
        elif operation == 'insert':
            return f"{components['action']} {components['table']} {components['fields']} {components['values']}{components['terminator']}"
        elif operation == 'update':
            return f"{components['action']} {components['table']} {components['set_clause']}{components['where_clause']}{components['terminator']}"
        elif operation == 'delete':
            return f"{components['action']} {components['from_clause']}{components['where_clause']}{components['terminator']}"

    def select(self, fields=None, condition=None):
        return self._assemble_query('select', fields=fields, condition=condition)

    def insert(self, data):
        return self._assemble_query('insert', data=data)

    def update(self, data, condition):
        return self._assemble_query('update', data=data, condition=condition)

    def delete(self, condition):
        return self._assemble_query('delete', condition=condition)

    def select_female_under_age(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.select(condition=condition)

    def select_by_age_range(self, min_age, max_age):
        condition = f"age BETWEEN {min_age} AND {max_age}"
        return self.select(condition=condition)
