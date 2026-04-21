def is_integer(x):
    return isinstance(x, int)

return list(filter(is_integer, values))
