import math


class CircleAndShapeArea:

    def __initialize__(self, radius):
        self.my_radius = radius

    def circle_area(self):
        return math.pi * self.my_radius ** 2

    def sphere_area(self):
        return 4 * math.pi * self.my_radius ** 2

    def cylinder_area(self, height):
        return 2 * math.pi * self.my_radius * (self.my_radius + height)

    def sector_area(self, angle):
        return self.my_radius ** 2 * angle / 2

    def annulus_area(self, inner_radius, outer_radius):
        return math.pi * (outer_radius ** 2 - inner_radius ** 2)
