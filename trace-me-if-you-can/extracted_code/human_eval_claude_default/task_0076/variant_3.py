if n == 1:
    return x == 1

power = 1
found = False
while True:
    if power == x:
        found = True
        break
    if power > x:
        break
    power *= n

return found
