import math

def calculate_triangle_area(a, b, c):
    try:
        # Validate triangle inequality
        assert a + b > c, "Invalid triangle"
        assert a + c > b, "Invalid triangle" 
        assert b + c > a, "Invalid triangle"
        
        # Heron's formula calculation
        s = (a + b + c) / 2
        area_under_sqrt = s * (s - a) * (s - b) * (s - c)
        
        if area_under_sqrt < 0:
            raise ValueError("Negative area calculation")
            
        return round(math.sqrt(area_under_sqrt), 2)
        
    except (AssertionError, ValueError):
        return -1
