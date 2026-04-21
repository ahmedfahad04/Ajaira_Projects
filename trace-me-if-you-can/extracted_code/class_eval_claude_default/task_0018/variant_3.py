class CamelCaseMap:
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[self._process_key(key)]

    def __setitem__(self, key, value):
        self._data[self._process_key(key)] = value

    def __delitem__(self, key):
        del self._data[self._process_key(key)]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def _process_key(self, key):
        if not isinstance(key, str):
            return key
        
        segments = key.split('_')
        if len(segments) == 1:
            return segments[0]
        
        return segments[0] + ''.join(segment.title() for segment in segments[1:])
