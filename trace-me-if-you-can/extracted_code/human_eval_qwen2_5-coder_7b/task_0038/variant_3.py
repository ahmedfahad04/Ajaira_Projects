def double_encode(text, func):
    return func(func(text))

return double_encode(s, encode_cyclic)
