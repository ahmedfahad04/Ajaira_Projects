class BisectionSolver:
    def __init__(self, tolerance=1e-10):
        self.tolerance = tolerance
    
    def expand_interval(self, xs, left, right):
        while poly(xs, left) * poly(xs, right) > 0:
            left, right = left * 2.0, right * 2.0
        return left, right
    
    def narrow_interval(self, xs, left, right):
        while right - left > self.tolerance:
            middle = (left + right) / 2.0
            if poly(xs, middle) * poly(xs, left) > 0:
                left = middle
            else:
                right = middle
        return left

solver = BisectionSolver()
expanded_left, expanded_right = solver.expand_interval(xs, -1., 1.)
return solver.narrow_interval(xs, expanded_left, expanded_right)
