import math


class AreaCalculator:

    def __init__(self, radius):
        self.radius = radius
        self._pi = math.pi
        self._radius_squared = radius ** 2

    def _get_base_circle_area(self):
        return self._pi * self._radius_squared

    def calculate_circle_area(self):
        return self._get_base_circle_area()

    def calculate_sphere_area(self):
        base_area = self._get_base_circle_area()
        return base_area * 4

    def calculate_cylinder_area(self, height):
        base_area = self._get_base_circle_area()
        lateral_area = 2 * self._pi * self.radius * height
        return 2 * base_area + lateral_area

    def calculate_sector_area(self, angle):
        return self._radius_squared * angle / 2

    def calculate_annulus_area(self, inner_radius, outer_radius):
        outer_area = self._pi * outer_radius ** 2
        inner_area = self._pi * inner_radius ** 2
        return outer_area - inner_area
