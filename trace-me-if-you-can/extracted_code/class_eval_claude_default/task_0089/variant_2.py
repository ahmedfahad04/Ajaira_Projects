import random
from collections import Counter


class TwentyFourPointGame:
    def __init__(self) -> None:
        self.nums = []

    def _generate_cards(self):
        for i in range(4):
            self.nums.append(random.randint(1, 9))
        assert len(self.nums) == 4

    def get_my_cards(self):
        self.nums = []
        self._generate_cards()
        return self.nums

    def answer(self, expression):
        if expression == 'pass':
            return self.get_my_cards()
        
        # Use Counter for more elegant counting
        expression_digits = [c for c in expression if c.isdigit() and int(c) in self.nums]
        expression_counter = Counter(expression_digits)
        nums_counter = Counter(str(num) for num in self.nums)
        
        # Check if digit usage matches exactly
        if expression_counter != nums_counter:
            return False
            
        return self.evaluate_expression(expression)

    def evaluate_expression(self, expression):
        try:
            result = eval(expression)
            return result == 24
        except Exception as e:
            return False
