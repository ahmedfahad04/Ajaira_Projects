class SQLQueryConstructor:

    @staticmethod
    def retrieve(table, fields='*', conditions=None):
        if fields != '*':
            fields = ', '.join(fields)
        query = f"SELECT {fields} FROM {table}"
        if conditions:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in conditions.items())
        return query

    @staticmethod
    def addRecord(table, info):
        keys = ', '.join(info.keys())
        values = ', '.join(f"'{v}'" for v in info.values())
        return f"INSERT INTO {table} ({keys}) VALUES ({values})"

    @staticmethod
    def purge(table, conditions=None):
        query = f"DELETE FROM {table}"
        if conditions:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in conditions.items())
        return query

    @staticmethod
    def modify(table, modifications, conditions=None):
        updates = ', '.join(f"{k}='{v}'" for k, v in modifications.items())
        query = f"UPDATE {table} SET {updates}"
        if conditions:
            query += " WHERE " + ' AND '.join(f"{k}='{v}'" for k, v in conditions.items())
        return query
