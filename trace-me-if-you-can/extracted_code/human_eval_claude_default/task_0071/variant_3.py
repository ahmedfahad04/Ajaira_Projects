def calculate_triangle_area(a, b, c):
    sides = [(a, b, c), (a, c, b), (b, c, a)]
    
    # Check if any pair sum is less than or equal to the third side
    if any(x + y <= z for x, y, z in sides):
        return -1
    
    half_perimeter = (a + b + c) * 0.5
    heron_product = half_perimeter
    
    for side in [a, b, c]:
        heron_product *= (half_perimeter - side)
    
    return round(heron_product ** 0.5, 2)
