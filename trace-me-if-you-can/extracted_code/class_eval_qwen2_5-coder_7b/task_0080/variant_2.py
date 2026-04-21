class QueryFormatter:

    @staticmethod
    def fetch(table, columns='*', criteria=None):
        if columns != '*':
            columns = ', '.join(columns)
        query = f"SELECT {columns} FROM {table}"
        if criteria:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in criteria.items())
        return query

    @staticmethod
    def insertData(table, data):
        keys = ', '.join(data.keys())
        values = ', '.join(f"'{v}'" for v in data.values())
        return f"INSERT INTO {table} ({keys}) VALUES ({values})"

    @staticmethod
    def remove(table, criteria=None):
        query = f"DELETE FROM {table}"
        if criteria:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in criteria.items())
        return query

    @staticmethod
    def edit(table, updates, criteria=None):
        update_str = ', '.join(f"{k}='{v}'" for k, v in updates.items())
        query = f"UPDATE {table} SET {update_str}"
        if criteria:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in criteria.items())
        return query
