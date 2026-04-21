def power_sequence(n):
    power = 1
    while True:
        yield power
        power *= n

if n == 1:
    return x == 1

for power in power_sequence(n):
    if power == x:
        return True
    if power > x:
        return False
