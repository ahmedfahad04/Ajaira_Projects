class SnakeToCamel:
    def __init__(self):
        self.data_store = {}

    def access_key(self, key):
        return self.data_store[self._snake_to_camel(key)]

    def store_key(self, key, value):
        self.data_store[self._snake_to_camel(key)] = value

    def remove_key(self, key):
        del self.data_store[self._snake_to_camel(key)]

    def list_keys(self):
        return iter(self.data_store)

    def count_keys(self):
        return len(self.data_store)

    @staticmethod
    def _snake_to_camel(key):
        return key[0] + ''.join(word.capitalize() for word in key.split('_')[1:])
