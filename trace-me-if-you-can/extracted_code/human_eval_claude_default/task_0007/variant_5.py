from itertools import compress

return list(compress(strings, (substring in s for s in strings)))
