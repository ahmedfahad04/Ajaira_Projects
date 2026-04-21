class SQLQueryBuilder:

    @staticmethod
    def select(table, columns='*', where=None):
        def build_query():
            yield "SELECT"
            yield columns if columns == '*' else ', '.join(columns)
            yield "FROM"
            yield table
            if where:
                yield "WHERE"
                yield ' AND '.join(f"{k}='{v}'" for k, v in where.items())
        
        return ' '.join(build_query())

    @staticmethod
    def insert(table, data):
        def build_query():
            yield "INSERT INTO"
            yield table
            yield f"({', '.join(data.keys())})"
            yield "VALUES"
            yield f"({', '.join(f\"'{v}'\" for v in data.values())})"
        
        return ' '.join(build_query())

    @staticmethod
    def delete(table, where=None):
        def build_query():
            yield "DELETE FROM"
            yield table
            if where:
                yield "WHERE"
                yield ' AND '.join(f"{k}='{v}'" for k, v in where.items())
        
        return ' '.join(build_query())

    @staticmethod
    def update(table, data, where=None):
        def build_query():
            yield "UPDATE"
            yield table
            yield "SET"
            yield ', '.join(f"{k}='{v}'" for k, v in data.items())
            if where:
                yield "WHERE"
                yield ' AND '.join(f"{k}='{v}'" for k, v in where.items())
        
        return ' '.join(build_query())
