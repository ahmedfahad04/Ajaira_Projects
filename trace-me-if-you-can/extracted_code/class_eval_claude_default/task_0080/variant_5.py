class SQLQueryBuilder:

    @staticmethod
    def select(table, columns='*', where=None):
        query_parts = {
            'base': f"SELECT {columns if columns == '*' else ', '.join(columns)} FROM {table}",
            'where': f" WHERE {' AND '.join(f\"{k}='{v}'\" for k, v in where.items())}" if where else ""
        }
        return query_parts['base'] + query_parts['where']

    @staticmethod
    def insert(table, data):
        query_components = {
            'statement': "INSERT INTO",
            'table': table,
            'columns': f"({', '.join(data.keys())})",
            'values_keyword': "VALUES",
            'values': f"({', '.join(f\"'{v}'\" for v in data.values())})"
        }
        return f"{query_components['statement']} {query_components['table']} {query_components['columns']} {query_components['values_keyword']} {query_components['values']}"

    @staticmethod
    def delete(table, where=None):
        base_query = f"DELETE FROM {table}"
        where_part = f" WHERE {' AND '.join(f\"{k}='{v}'\" for k, v in where.items())}" if where else ""
        return base_query + where_part

    @staticmethod
    def update(table, data, where=None):
        query_elements = [
            f"UPDATE {table}",
            f"SET {', '.join(f\"{k}='{v}'\" for k, v in data.items())}"
        ]
        
        if where:
            query_elements.append(f"WHERE {' AND '.join(f\"{k}='{v}'\" for k, v in where.items())}")
        
        return ' '.join(query_elements)
