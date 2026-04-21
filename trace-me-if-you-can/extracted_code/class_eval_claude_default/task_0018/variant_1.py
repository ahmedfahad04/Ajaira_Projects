class CamelCaseMap:
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        normalized_key = self._normalize_key(key)
        return self._data[normalized_key]

    def __setitem__(self, key, value):
        normalized_key = self._normalize_key(key)
        self._data[normalized_key] = value

    def __delitem__(self, key):
        normalized_key = self._normalize_key(key)
        del self._data[normalized_key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def _normalize_key(self, key):
        return self._snake_to_camel(key) if isinstance(key, str) else key

    def _snake_to_camel(self, snake_str):
        components = snake_str.split('_')
        return components[0] + ''.join(x.capitalize() for x in components[1:])
