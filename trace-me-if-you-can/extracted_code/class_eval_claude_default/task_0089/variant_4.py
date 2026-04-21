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
        
        # Create multiset representation of available numbers
        available_multiset = {}
        for num in self.nums:
            key = str(num)
            available_multiset[key] = available_multiset.get(key, 0) + 1
        
        # Create multiset representation of used digits
        used_multiset = {}
        for char in expression:
            if char.isdigit() and int(char) in self.nums:
                used_multiset[char] = used_multiset.get(char, 0) + 1
        
        # Multiset equality check
        multisets_equal = (len(available_multiset) == len(used_multiset) and
                          all(available_multiset.get(k, 0) == used_multiset.get(k, 0) 
                              for k in set(available_multiset.keys()) | set(used_multiset.keys())))
        
        return multisets_equal and self.evaluate_expression(expression)

    def evaluate_expression(self, expression):
        try:
            if eval(expression) == 24:
                return True
            else:
                return False
        except Exception as e:
            return False
