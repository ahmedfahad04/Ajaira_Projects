import random


class TwentyFourPointGame:
    def __init__(self) -> None:
        self.nums = []

    def _generate_cards(self):
        self.nums = [random.randint(1, 9) for _ in range(4)]
        assert len(self.nums) == 4

    def get_my_cards(self):
        self.nums = []
        self._generate_cards()
        return self.nums

    def answer(self, expression):
        if expression == 'pass':
            return self.get_my_cards()
        
        # Extract digits and count occurrences
        digit_counts = {}
        for char in expression:
            if char.isdigit() and int(char) in self.nums:
                digit_counts[char] = digit_counts.get(char, 0) + 1

        # Check if all required numbers are used exactly once
        available_nums = [str(num) for num in self.nums]
        for digit, count in digit_counts.items():
            for _ in range(count):
                if digit in available_nums:
                    available_nums.remove(digit)
                else:
                    return False
        
        # All numbers must be used
        if available_nums:
            return False
            
        return self.evaluate_expression(expression)

    def evaluate_expression(self, expression):
        try:
            return eval(expression) == 24
        except Exception:
            return False
