class SQLQueryBuilder:
    
    @staticmethod
    def _build_where_clause(where):
        return ' AND '.join(f"{k}='{v}'" for k, v in where.items()) if where else None
    
    @staticmethod
    def select(table, columns='*', where=None):
        column_str = ', '.join(columns) if columns != '*' else '*'
        query = f"SELECT {column_str} FROM {table}"
        where_clause = SQLQueryBuilder._build_where_clause(where)
        return query + (f" WHERE {where_clause}" if where_clause else "")

    @staticmethod
    def insert(table, data):
        keys = ', '.join(data.keys())
        values = ', '.join(f"'{v}'" for v in data.values())
        return f"INSERT INTO {table} ({keys}) VALUES ({values})"

    @staticmethod
    def delete(table, where=None):
        query = f"DELETE FROM {table}"
        where_clause = SQLQueryBuilder._build_where_clause(where)
        return query + (f" WHERE {where_clause}" if where_clause else "")

    @staticmethod
    def update(table, data, where=None):
        update_str = ', '.join(f"{k}='{v}'" for k, v in data.items())
        query = f"UPDATE {table} SET {update_str}"
        where_clause = SQLQueryBuilder._build_where_clause(where)
        return query + (f" WHERE {where_clause}" if where_clause else "")
