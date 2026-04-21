import itertools
numbers = itertools.islice(itertools.count(0), n + 1)
return ' '.join(str(num) for num in numbers)
