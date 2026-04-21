# Using bitwise operations for optimization when both numbers are even
while b:
    if a & 1 == 0 and b & 1 == 0:
        # Both even, divide by 2
        a >>= 1
        b >>= 1
        shift = 1
    else:
        shift = 0
        break
    
while b:
    a, b = b, a % b
    
return a << shift if 'shift' in locals() and shift else a
