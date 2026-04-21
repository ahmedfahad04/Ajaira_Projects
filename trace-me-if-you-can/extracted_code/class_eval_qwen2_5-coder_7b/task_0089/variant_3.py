import random

class MathPuzzle:
    def __init__(self):
        self.nums = []

    def _create_nums(self):
        for _ in range(4):
            self.nums.append(random.randint(1, 9))
        assert len(self.nums) == 4

    def refresh_nums(self):
        self.nums = []
        self._create_nums()
        return self.nums

    def assess(self, equation):
        if equation == 'pass':
            return self.refresh_nums()
        counter = {}
        for char in equation:
            if char.isdigit() and int(char) in self.nums:
                counter[char] = counter.get(char, 0) + 1

        used_counter = counter.copy()

        for num in self.nums:
            if used_counter.get(str(num), -100) != -100 and used_counter[str(num)] > 0:
                used_counter[str(num)] -= 1
            else:
                return False

        if all(count == 0 for count in used_counter.values()):
            return self.check_result(equation)
        else:
            return False

    def check_result(self, equation):
        try:
            result = eval(equation)
            return result == 24
        except:
            return False
