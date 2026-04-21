def digit_generator(num, base):
    while num > 0:
        yield str(num % base)
        num //= base

ret = ''.join(reversed(list(digit_generator(x, base))))
return ret
