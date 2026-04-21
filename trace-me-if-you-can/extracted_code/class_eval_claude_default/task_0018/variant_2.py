class CamelCaseMap:
    def __init__(self):
        self._storage = {}
        self._key_transformer = lambda k: k if not isinstance(k, str) else self._camel_case_transform(k)

    def __getitem__(self, key):
        return self._storage[self._key_transformer(key)]

    def __setitem__(self, key, value):
        self._storage[self._key_transformer(key)] = value

    def __delitem__(self, key):
        del self._storage[self._key_transformer(key)]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def _camel_case_transform(self, text):
        tokens = text.split('_')
        return tokens[0] + ''.join(token.title() for token in tokens[1:])
