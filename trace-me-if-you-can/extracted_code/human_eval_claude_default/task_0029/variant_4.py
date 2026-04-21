from itertools import compress
return list(compress(strings, [x.startswith(prefix) for x in strings]))
