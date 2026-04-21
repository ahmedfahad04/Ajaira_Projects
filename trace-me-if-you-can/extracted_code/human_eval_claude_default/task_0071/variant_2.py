import math

def calculate_triangle_area(a, b, c):
    sides = [a, b, c]
    sides.sort()
    
    # Triangle inequality: sum of two smaller sides must be greater than largest
    if sides[0] + sides[1] <= sides[2]:
        return -1
    
    s = sum(sides) / 2
    discriminant = s * (s - a) * (s - b) * (s - c)
    area = math.sqrt(discriminant)
    
    return round(area, 2)
