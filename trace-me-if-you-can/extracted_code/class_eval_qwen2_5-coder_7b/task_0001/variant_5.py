import math

class AreaComputer:

    def __init__(self, dimension):
        self.dimension = dimension

    def get_circle_surface(self):
        return math.pi * self.dimension ** 2

    def get_sphere_surface(self):
        return 4 * math.pi * self.dimension ** 2

    def get_cylinder_surface(self, height):
        return 2 * math.pi * self.dimension * (self.dimension + height)

    def get_sector_surface(self, angle):
        return self.dimension ** 2 * angle / 2

    def get_annulus_surface(self, inner_dim, outer_dim):
        return math.pi * (outer_dim ** 2 - inner_dim ** 2)
