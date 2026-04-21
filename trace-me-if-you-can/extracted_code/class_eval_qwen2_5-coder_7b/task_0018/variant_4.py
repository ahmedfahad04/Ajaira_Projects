class CaseConverter:
    def __init__(self):
        self.data = {}

    def get_value(self, key):
        return self.data[self._convert_to_camel(key)]

    def set_value(self, key, value):
        self.data[self._convert_to_camel(key)] = value

    def delete_value(self, key):
        del self.data[self._convert_to_camel(key)]

    def keys(self):
        return iter(self.data)

    def size(self):
        return len(self.data)

    @staticmethod
    def _convert_to_camel(key):
        parts = key.split('_')
        return parts[0] + ''.join(part.capitalize() for part in parts[1:])
