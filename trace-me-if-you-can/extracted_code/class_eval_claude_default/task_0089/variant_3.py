import random


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

    def _validate_digit_usage(self, expression):
        """Validate that expression uses exactly the available digits"""
        available_digits = sorted([str(num) for num in self.nums])
        used_digits = sorted([c for c in expression if c.isdigit() and int(c) in self.nums])
        return available_digits == used_digits

    def _is_valid_expression(self, expression):
        """Check if expression evaluates to 24"""
        try:
            return eval(expression) == 24
        except Exception:
            return False

    def answer(self, expression):
        if expression == 'pass':
            return self.get_my_cards()
        
        return (self._validate_digit_usage(expression) and 
                self._is_valid_expression(expression))

    def evaluate_expression(self, expression):
        try:
            if eval(expression) == 24:
                return True
            else:
                return False
        except Exception as e:
            return False
