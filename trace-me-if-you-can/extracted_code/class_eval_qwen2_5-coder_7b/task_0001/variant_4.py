from math import pi


class GeometricAreaCalculator:

    def __init__(self, r):
        self.radius = r

    def calculate_circle(self):
        return pi * self.radius ** 2

    def calculate_sphere(self):
        return 4 * pi * self.radius ** 2

    def calculate_cylinder(self, h):
        return 2 * pi * self.radius * (self.radius + h)

    def calculate_sector(self, a):
        return self.radius ** 2 * a / 2

    def calculate_annulus(self, inner_r, outer_r):
        return pi * (outer_r ** 2 - inner_r ** 2)
