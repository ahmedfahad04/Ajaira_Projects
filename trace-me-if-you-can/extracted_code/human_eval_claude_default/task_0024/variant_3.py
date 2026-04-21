# Version 3: Generator with next()
def divisor_generator(num):
    for i in reversed(range(1, num)):
        if num % i == 0:
            yield i

return next(divisor_generator(n), 1)
