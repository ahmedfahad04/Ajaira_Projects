def compute_area_triangle(side_a, side_b, side_c):
    if side_a + side_b <= side_c or side_a + side_c <= side_b or side_b + side_c <= side_a:
        return -1
    
    semi_perimeter = (side_a + side_b + side_c) / 2
    area = (semi_perimeter * (semi_perimeter - side_a) * (semi_perimeter - side_b) * (semi_perimeter - side_c)) ** 0.5
    return round(area, 2)
