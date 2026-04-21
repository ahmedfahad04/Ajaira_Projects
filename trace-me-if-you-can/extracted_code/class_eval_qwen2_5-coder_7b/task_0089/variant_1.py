import random

class NumberGame:
    def __init__(self):
        self.cards = []

    def _draw_numbers(self):
        for _ in range(4):
            self.cards.append(random.randint(1, 9))
        assert len(self.cards) == 4

    def fetch_cards(self):
        self.cards = []
        self._draw_numbers()
        return self.cards

    def solve(self, formula):
        if formula == 'pass':
            return self.fetch_cards()
        number_tracker = {}
        for char in formula:
            if char.isdigit() and int(char) in self.cards:
                number_tracker[char] = number_tracker.get(char, 0) + 1

        used_numbers = number_tracker.copy()

        for card in self.cards:
            if used_numbers.get(str(card), -100) != -100 and used_numbers[str(card)] > 0:
                used_numbers[str(card)] -= 1
            else:
                return False

        if all(count == 0 for count in used_numbers.values()):
            return self.check_solution(formula)
        else:
            return False

    def check_solution(self, formula):
        try:
            result = eval(formula)
            return result == 24
        except:
            return False
