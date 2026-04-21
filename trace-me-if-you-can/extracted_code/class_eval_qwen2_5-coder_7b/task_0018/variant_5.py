class CamelCaseHolder:
    def __init__(self):
        self.contents = {}

    def fetch(self, key):
        return self.contents[self._toCamel(key)]

    def store(self, key, value):
        self.contents[self._toCamel(key)] = value

    def erase(self, key):
        del self.contents[self._toCamel(key)]

    def iterateContents(self):
        return iter(self.contents)

    def countContents(self):
        return len(self.contents)

    def _toCamel(self, key):
        return ''.join(word.capitalize() for word in key.split('_'))
