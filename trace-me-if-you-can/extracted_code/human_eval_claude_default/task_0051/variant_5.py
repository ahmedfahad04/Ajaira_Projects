from functools import reduce
return reduce(lambda acc, char: acc + char if char.lower() not in "aeiou" else acc, text, "")
