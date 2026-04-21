class QueryEngine:

    @staticmethod
    def selectData(table, fields='*', conditions=None):
        if fields != '*':
            fields = ', '.join(fields)
        query = f"SELECT {fields} FROM {table}"
        if conditions:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in conditions.items())
        return query

    @staticmethod
    def insertRow(table, data):
        keys = ', '.join(data.keys())
        values = ', '.join(f"'{v}'" for v in data.values())
        return f"INSERT INTO {table} ({keys}) VALUES ({values})"

    @staticmethod
    def removeRow(table, conditions=None):
        query = f"DELETE FROM {table}"
        if conditions:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in conditions.items())
        return query

    @staticmethod
    def updateRow(table, changes, conditions=None):
        update_str = ', '.join(f"{k}='{v}'" for k, v in changes.items())
        query = f"UPDATE {table} SET {update_str}"
        if conditions:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in conditions.items())
        return query
