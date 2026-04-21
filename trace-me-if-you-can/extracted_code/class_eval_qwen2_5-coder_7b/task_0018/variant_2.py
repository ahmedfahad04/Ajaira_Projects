class UpperCamelCaseStorage:
    def __init__(self):
        self._map = {}

    def getItem(self, key):
        return self._map[self._transformKey(key)]

    def setItem(self, key, value):
        self._map[self._transformKey(key)] = value

    def deleteItem(self, key):
        del self._map[self._transformKey(key)]

    def iterateItems(self):
        return iter(self._map)

    def getItemsCount(self):
        return len(self._map)

    def _transformKey(self, key):
        return key.replace('_', ' ').title().replace(' ', '')
