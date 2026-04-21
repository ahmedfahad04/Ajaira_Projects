import random

class NumberChallenge:
    def __init__(self):
        self.values = []

    def _generate_numbers(self):
        for _ in range(4):
            self.values.append(random.randint(1, 9))
        assert len(self.values) == 4

    def get_numbers(self):
        self.values = []
        self._generate_numbers()
        return self.values

    def process(self, equation):
        if equation == 'pass':
            return self.get_numbers()
        count_map = {}
        for char in equation:
            if char.isdigit() and int(char) in self.values:
                count_map[char] = count_map.get(char, 0) + 1

        used_counts = count_map.copy()

        for value in self.values:
            if used_counts.get(str(value), -100) != -100 and used_counts[str(value)] > 0:
                used_counts[str(value)] -= 1
            else:
                return False

        if all(count == 0 for count in used_counts.values()):
            return self.validate_equation(equation)
        else:
            return False

    def validate_equation(self, equation):
        try:
            outcome = eval(equation)
            return outcome == 24
        except:
            return False
