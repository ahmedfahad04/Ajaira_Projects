class SQLQueryBuilder:

    @staticmethod
    def select(table, columns='*', where=None):
        query_template = "SELECT {columns} FROM {table}{where_clause}"
        column_str = columns if columns == '*' else ', '.join(columns)
        where_clause = ""
        if where:
            where_clause = " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in where.items())
        
        return query_template.format(
            columns=column_str,
            table=table,
            where_clause=where_clause
        )

    @staticmethod
    def insert(table, data):
        template = "INSERT INTO {table} ({keys}) VALUES ({values})"
        return template.format(
            table=table,
            keys=', '.join(data.keys()),
            values=', '.join(f"'{v}'" for v in data.values())
        )

    @staticmethod
    def delete(table, where=None):
        template = "DELETE FROM {table}{where_clause}"
        where_clause = ""
        if where:
            where_clause = " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in where.items())
        
        return template.format(table=table, where_clause=where_clause)

    @staticmethod
    def update(table, data, where=None):
        template = "UPDATE {table} SET {updates}{where_clause}"
        where_clause = ""
        if where:
            where_clause = " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in where.items())
        
        return template.format(
            table=table,
            updates=', '.join(f"{k}='{v}'" for k, v in data.items()),
            where_clause=where_clause
        )
