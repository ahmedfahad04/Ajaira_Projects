from itertools import islice
even_chars = islice(s, 0, None, 2)
return sum(map(lambda char: char in "AEIOU", even_chars))
