from math import pi


class AreaComputer:

    def __init__(self, radius):
        self.rad = radius

    def get_circle_area(self):
        return pi * self.rad ** 2

    def get_sphere_area(self):
        return 4 * pi * self.rad ** 2

    def get_cylinder_area(self, height):
        return 2 * pi * self.rad * (self.rad + height)

    def get_sector_area(self, angle):
        return self.rad ** 2 * angle / 2

    def get_annulus_area(self, inner_rad, outer_rad):
        return pi * (outer_rad ** 2 - inner_rad ** 2)
