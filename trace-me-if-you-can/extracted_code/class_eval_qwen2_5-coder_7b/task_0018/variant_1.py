class CamelCaseDict:
    def __init__(self):
        self.storage = {}

    def get_item(self, key):
        return self.storage[self._to_camel_case(key)]

    def set_item(self, key, value):
        self.storage[self._to_camel_case(key)] = value

    def delete_item(self, key):
        del self.storage[self._to_camel_case(key)]

    def iterate(self):
        return iter(self.storage)

    def get_length(self):
        return len(self.storage)

    def _to_camel_case(self, key):
        parts = key.split('_')
        return parts[0] + ''.join(part.title() for part in parts[1:])
