import math


class ShapeAreaCalculator:

    def __init__(self, radius):
        self.diameter = 2 * radius
        self.pi = math.pi

    def compute_circle_area(self):
        return self.pi * self.diameter ** 2 / 4

    def compute_sphere_area(self):
        return 4 * self.pi * self.diameter ** 2 / 16

    def compute_cylinder_area(self, height):
        return self.pi * self.diameter * (self.diameter / 2 + height)

    def compute_sector_area(self, angle):
        return self.diameter ** 2 * angle / 8

    def compute_annulus_area(self, inner_radius, outer_radius):
        return self.pi * (outer_radius ** 2 - inner_radius ** 2)
