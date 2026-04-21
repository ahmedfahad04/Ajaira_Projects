import random

class FourNumberGame:
    def __init__(self):
        self.numbers = []

    def _generate_set(self):
        for _ in range(4):
            self.numbers.append(random.randint(1, 9))
        assert len(self.numbers) == 4

    def get_set(self):
        self.numbers = []
        self._generate_set()
        return self.numbers

    def solve_problem(self, formula):
        if formula == 'pass':
            return self.get_set()
        digit_map = {}
        for char in formula:
            if char.isdigit() and int(char) in self.numbers:
                digit_map[char] = digit_map.get(char, 0) + 1

        used_digits = digit_map.copy()

        for number in self.numbers:
            if used_digits.get(str(number), -100) != -100 and used_digits[str(number)] > 0:
                used_digits[str(number)] -= 1
            else:
                return False

        if all(count == 0 for count in used_digits.values()):
            return self.check_solution(formula)
        else:
            return False

    def check_solution(self, formula):
        try:
            result = eval(formula)
            return result == 24
        except:
            return False
