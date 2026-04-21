import math


class AreaCalculator:
    
    @staticmethod
    def calculate_circle_area(radius):
        return math.pi * radius ** 2
    
    @staticmethod
    def calculate_sphere_area(radius):
        return 4 * math.pi * radius ** 2
    
    @staticmethod
    def calculate_cylinder_area(radius, height):
        return 2 * math.pi * radius * (radius + height)
    
    @staticmethod
    def calculate_sector_area(radius, angle):
        return radius ** 2 * angle / 2
    
    @staticmethod
    def calculate_annulus_area(inner_radius, outer_radius):
        return math.pi * (outer_radius ** 2 - inner_radius ** 2)
    
    def __init__(self, radius):
        self.radius = radius
        self.calculate_circle_area = lambda: self.calculate_circle_area(self.radius)
        self.calculate_sphere_area = lambda: self.calculate_sphere_area(self.radius)
        self.calculate_cylinder_area = lambda height: self.calculate_cylinder_area(self.radius, height)
        self.calculate_sector_area = lambda angle: self.calculate_sector_area(self.radius, angle)
