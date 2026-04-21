class CamelCaseMap:
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[self._get_camel_key(key)]

    def __setitem__(self, key, value):
        self._data[self._get_camel_key(key)] = value

    def __delitem__(self, key):
        del self._data[self._get_camel_key(key)]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def _get_camel_key(self, original_key):
        if not isinstance(original_key, str):
            return original_key
        
        parts = original_key.split('_')
        camel_key = parts[0]
        
        for part in parts[1:]:
            camel_key += part.title()
            
        return camel_key
