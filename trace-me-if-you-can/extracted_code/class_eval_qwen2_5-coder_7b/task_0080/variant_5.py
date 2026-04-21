class SQLCommandGenerator:

    @staticmethod
    def fetch_data(table, columns='*', conditions=None):
        columns = ', '.join(columns) if columns != '*' else '*'
        base_query = f"SELECT {columns} FROM {table}"
        if conditions:
            conditions_str = ' AND '.join(f"{key}='{value}'" for key, value in conditions.items())
            return base_query + " WHERE " + conditions_str
        return base_query

    @staticmethod
    def insert_record(table, data):
        keys = ', '.join(data.keys())
        values = ', '.join(f"'{value}'" for value in data.values())
        return f"INSERT INTO {table} ({keys}) VALUES ({values})"

    @staticmethod
    def remove_record(table, conditions=None):
        base_query = f"DELETE FROM {table}"
        if conditions:
            conditions_str = ' AND '.join(f"{key}='{value}'" for key, value in conditions.items())
            return base_query + " WHERE " + conditions_str
        return base_query

    @staticmethod
    def modify_record(table, data, conditions=None):
        update_clause = ', '.join(f"{key}='{value}'" for key, value in data.items())
        base_query = f"UPDATE {table} SET {update_clause}"
        if conditions:
            conditions_str = ' AND '.join(f"{key}='{value}'" for key, value in conditions.items())
            return base_query + " WHERE " + conditions_str
        return base_query
