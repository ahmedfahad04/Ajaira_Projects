from abc import ABC, abstractmethod

class QueryBuilder(ABC):
    @abstractmethod
    def build(self, table_name, **kwargs):
        pass

class SelectBuilder(QueryBuilder):
    def build(self, table_name, fields=None, condition=None):
        field_str = "*" if fields is None else ", ".join(fields)
        sql = f"SELECT {field_str} FROM {table_name}"
        if condition:
            sql += f" WHERE {condition}"
        return sql + ";"

class InsertBuilder(QueryBuilder):
    def build(self, table_name, data):
        fields = ", ".join(data.keys())
        values = ", ".join([f"'{value}'" for value in data.values()])
        return f"INSERT INTO {table_name} ({fields}) VALUES ({values});"

class UpdateBuilder(QueryBuilder):
    def build(self, table_name, data, condition):
        set_clause = ", ".join([f"{field} = '{value}'" for field, value in data.items()])
        return f"UPDATE {table_name} SET {set_clause} WHERE {condition};"

class DeleteBuilder(QueryBuilder):
    def build(self, table_name, condition):
        return f"DELETE FROM {table_name} WHERE {condition};"

class SQLGenerator:
    def __init__(self, table_name):
        self.table_name = table_name
        self.builders = {
            'select': SelectBuilder(),
            'insert': InsertBuilder(),
            'update': UpdateBuilder(),
            'delete': DeleteBuilder()
        }

    def select(self, fields=None, condition=None):
        return self.builders['select'].build(self.table_name, fields=fields, condition=condition)

    def insert(self, data):
        return self.builders['insert'].build(self.table_name, data=data)

    def update(self, data, condition):
        return self.builders['update'].build(self.table_name, data=data, condition=condition)

    def delete(self, condition):
        return self.builders['delete'].build(self.table_name, condition=condition)

    def select_female_under_age(self, age):
        condition = f"age < {age} AND gender = 'female'"
        return self.select(condition=condition)

    def select_by_age_range(self, min_age, max_age):
        condition = f"age BETWEEN {min_age} AND {max_age}"
        return self.select(condition=condition)
