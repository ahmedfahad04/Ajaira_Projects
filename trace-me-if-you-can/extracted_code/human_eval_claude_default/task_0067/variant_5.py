# Variant 5: While loop with iterator approach
tokens = iter(s.split(' '))
digit_sum = 0
try:
    while True:
        token = next(tokens)
        if token.isdigit():
            digit_sum += int(token)
except StopIteration:
    pass
return n - digit_sum
