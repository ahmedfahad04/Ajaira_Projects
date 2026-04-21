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

    def answer(self, expression):
        if expression == 'pass':
            return self.get_my_cards()
        
        # Filter valid digits from expression
        valid_digits = [c for c in expression if c.isdigit() and int(c) in self.nums]
        
        # Early exit if wrong number of digits
        if len(valid_digits) != 4:
            return False
        
        # Sort both lists for comparison
        expected_digits = sorted([str(num) for num in self.nums])
        actual_digits = sorted(valid_digits)
        
        # Check exact match and evaluate
        digits_match = expected_digits == actual_digits
        expression_valid = self._safe_evaluate(expression)
        
        return digits_match and expression_valid

    def _safe_evaluate(self, expression):
        """Safely evaluate expression and check if equals 24"""
        try:
            return eval(expression) == 24
        except Exception:
            return False

    def evaluate_expression(self, expression):
        try:
            if eval(expression) == 24:
                return True
            else:
                return False
        except Exception as e:
            return False
