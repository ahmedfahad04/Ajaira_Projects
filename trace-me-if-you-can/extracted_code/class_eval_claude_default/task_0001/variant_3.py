import math


class AreaCalculator:

    def __init__(self, radius):
        self.radius = radius
        self._formulas = {
            'circle': lambda r: math.pi * r ** 2,
            'sphere': lambda r: 4 * math.pi * r ** 2,
            'cylinder': lambda r, h: 2 * math.pi * r * (r + h),
            'sector': lambda r, a: r ** 2 * a / 2,
            'annulus': lambda inner, outer: math.pi * (outer ** 2 - inner ** 2)
        }

    def calculate_circle_area(self):
        return self._formulas['circle'](self.radius)

    def calculate_sphere_area(self):
        return self._formulas['sphere'](self.radius)

    def calculate_cylinder_area(self, height):
        return self._formulas['cylinder'](self.radius, height)

    def calculate_sector_area(self, angle):
        return self._formulas['sector'](self.radius, angle)

    def calculate_annulus_area(self, inner_radius, outer_radius):
        return self._formulas['annulus'](inner_radius, outer_radius)
