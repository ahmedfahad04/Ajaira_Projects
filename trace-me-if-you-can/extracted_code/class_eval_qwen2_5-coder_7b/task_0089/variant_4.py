import random

class NumberGame24:
    def __init__(self):
        self.numbers = []

    def _draw(self):
        for _ in range(4):
            self.numbers.append(random.randint(1, 9))
        assert len(self.numbers) == 4

    def get_numbers(self):
        self.numbers = []
        self._draw()
        return self.numbers

    def evaluate(self, formula):
        if formula == 'pass':
            return self.get_numbers()
        digit_tracker = {}
        for char in formula:
            if char.isdigit() and int(char) in self.numbers:
                digit_tracker[char] = digit_tracker.get(char, 0) + 1

        used_digits = digit_tracker.copy()

        for number in self.numbers:
            if used_digits.get(str(number), -100) != -100 and used_digits[str(number)] > 0:
                used_digits[str(number)] -= 1
            else:
                return False

        if all(count == 0 for count in used_digits.values()):
            return self.check_formula(formula)
        else:
            return False

    def check_formula(self, formula):
        try:
            result = eval(formula)
            return result == 24
        except:
            return False
