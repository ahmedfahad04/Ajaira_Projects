# Functional approach using a helper function
def step(pair):
    x, y = pair
    return (y, x % y) if y else (x, 0)

current = (a, b)
while current[1]:
    current = step(current)
return current[0]
