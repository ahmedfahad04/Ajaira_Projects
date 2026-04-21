def is_integer(value):
    return isinstance(value, int)

return [x for x in values if is_integer(x)]
