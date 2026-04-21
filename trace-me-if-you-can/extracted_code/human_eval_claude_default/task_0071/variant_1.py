# Variant 1: Early validation with explicit triangle inequality checks
def calculate_triangle_area(a, b, c):
    # Check triangle inequality theorem
    if not (a + b > c and a + c > b and b + c > a):
        return -1
    
    # Calculate semi-perimeter
    semi_perimeter = (a + b + c) / 2
    
    # Apply Heron's formula
    area_squared = semi_perimeter * (semi_perimeter - a) * (semi_perimeter - b) * (semi_perimeter - c)
    area = area_squared ** 0.5
    
    return round(area, 2)
