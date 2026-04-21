from functools import wraps

class CamelCaseMap:
    def __init__(self):
        self._data = {}

    def _key_converter(func):
        @wraps(func)
        def wrapper(self, key, *args, **kwargs):
            converted_key = self._transform_to_camel(key) if isinstance(key, str) else key
            return func(self, converted_key, *args, **kwargs)
        return wrapper

    @_key_converter
    def __getitem__(self, key):
        return self._data[key]

    @_key_converter
    def __setitem__(self, key, value):
        self._data[key] = value

    @_key_converter
    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def _transform_to_camel(self, key):
        words = key.split('_')
        return words[0] + ''.join(word.title() for word in words[1:])
