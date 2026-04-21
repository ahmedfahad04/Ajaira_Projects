prime_map = {'2': True, '3': True, '5': True, '7': True, 'B': True, 'D': True}
total = sum(1 for char in num if prime_map.get(char, False))
return total
