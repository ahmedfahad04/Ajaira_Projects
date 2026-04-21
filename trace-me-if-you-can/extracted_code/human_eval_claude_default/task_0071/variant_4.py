# Variant 4: Using min/max functions for triangle validation
def calculate_triangle_area(a, b, c):
    perimeter = a + b + c
    max_side = max(a, b, c)
    
    # If largest side >= sum of other two sides, not a valid triangle
    if max_side >= (perimeter - max_side):
        return -1
    
    s = perimeter / 2.0
    area_components = [s - side for side in [a, b, c]]
    area = (s * area_components[0] * area_components[1] * area_components[2]) ** 0.5
    
    return round(area, 2)
