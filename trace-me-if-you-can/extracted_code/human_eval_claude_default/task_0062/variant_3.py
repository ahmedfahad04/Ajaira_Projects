def multiply_indexed(xs):
    return list(map(lambda pair: pair[0] * pair[1], enumerate(xs)))[1:]
return multiply_indexed(xs)
