def calculate_triangle_area(x, y, z):
    if x + y <= z or x + z <= y or y + z <= x:
        return -1
    
    s = (x + y + z) / 2
    area = (s * (s - x) * (s - y) * (s - z)) ** 0.5
    return round(area, 2)
