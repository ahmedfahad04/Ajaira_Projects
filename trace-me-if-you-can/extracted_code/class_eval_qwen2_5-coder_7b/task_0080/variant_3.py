class SQLStatementBuilder:

    @staticmethod
    def get(table, columns='*', filters=None):
        if columns != '*':
            columns = ', '.join(columns)
        query = f"SELECT {columns} FROM {table}"
        if filters:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in filters.items())
        return query

    @staticmethod
    def insertRecord(table, data):
        keys = ', '.join(data.keys())
        values = ', '.join(f"'{v}'" for v in data.values())
        return f"INSERT INTO {table} ({keys}) VALUES ({values})"

    @staticmethod
    def deleteRecord(table, filters=None):
        query = f"DELETE FROM {table}"
        if filters:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in filters.items())
        return query

    @staticmethod
    def updateRecord(table, data, filters=None):
        update_str = ', '.join(f"{k}='{v}'" for k, v in data.items())
        query = f"UPDATE {table} SET {update_str}"
        if filters:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in filters.items())
        return query
