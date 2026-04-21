class SQLQueryBuilder:

    @staticmethod
    def select(table, columns='*', where=None):
        parts = ["SELECT"]
        parts.append(', '.join(columns) if columns != '*' else '*')
        parts.extend(["FROM", table])
        
        if where:
            parts.append("WHERE")
            parts.append(' AND '.join(f"{k}='{v}'" for k, v in where.items()))
        
        return ' '.join(parts)

    @staticmethod
    def insert(table, data):
        return ' '.join([
            "INSERT INTO",
            table,
            f"({', '.join(data.keys())})",
            "VALUES",
            f"({', '.join(f\"'{v}'\" for v in data.values())})"
        ])

    @staticmethod
    def delete(table, where=None):
        parts = ["DELETE FROM", table]
        if where:
            parts.extend(["WHERE", ' AND '.join(f"{k}='{v}'" for k, v in where.items())])
        return ' '.join(parts)

    @staticmethod
    def update(table, data, where=None):
        parts = ["UPDATE", table, "SET", ', '.join(f"{k}='{v}'" for k, v in data.items())]
        if where:
            parts.extend(["WHERE", ' AND '.join(f"{k}='{v}'" for k, v in where.items())])
        return ' '.join(parts)
