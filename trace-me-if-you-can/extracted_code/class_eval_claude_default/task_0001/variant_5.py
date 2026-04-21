import math


class AreaCalculator:

    def __init__(self, radius):
        self.radius = radius

    def _compute_pi_r_squared(self, r=None):
        effective_radius = r if r is not None else self.radius
        return math.pi * effective_radius * effective_radius

    def _compute_circumference(self, r=None):
        effective_radius = r if r is not None else self.radius
        return 2 * math.pi * effective_radius

    def calculate_circle_area(self):
        return self._compute_pi_r_squared()

    def calculate_sphere_area(self):
        return 4 * self._compute_pi_r_squared()

    def calculate_cylinder_area(self, height):
        top_bottom = 2 * self._compute_pi_r_squared()
        side = self._compute_circumference() * height
        return top_bottom + side

    def calculate_sector_area(self, angle):
        full_circle = self._compute_pi_r_squared()
        return full_circle * angle / (2 * math.pi)

    def calculate_annulus_area(self, inner_radius, outer_radius):
        outer_area = self._compute_pi_r_squared(outer_radius)
        inner_area = self._compute_pi_r_squared(inner_radius)
        return outer_area - inner_area
