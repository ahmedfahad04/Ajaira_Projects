import math


class AreaCalculator:

    def __init__(self, radius):
        self._radius = radius
        self._circle_area_cache = None
        self._sphere_area_cache = None

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self._circle_area_cache = None
        self._sphere_area_cache = None

    def calculate_circle_area(self):
        if self._circle_area_cache is None:
            self._circle_area_cache = math.pi * self._radius ** 2
        return self._circle_area_cache

    def calculate_sphere_area(self):
        if self._sphere_area_cache is None:
            self._sphere_area_cache = 4 * self.calculate_circle_area()
        return self._sphere_area_cache

    def calculate_cylinder_area(self, height):
        circle_area = self.calculate_circle_area()
        circumference = 2 * math.pi * self._radius
        return 2 * circle_area + circumference * height

    def calculate_sector_area(self, angle):
        return self._radius ** 2 * angle * 0.5

    def calculate_annulus_area(self, inner_radius, outer_radius):
        return math.pi * (outer_radius * outer_radius - inner_radius * inner_radius)
