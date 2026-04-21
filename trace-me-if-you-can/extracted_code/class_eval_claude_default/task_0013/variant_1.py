from collections import defaultdict

class BookManagement:
    def __init__(self):
        self.inventory = defaultdict(int)

    def add_book(self, title, quantity=1):
        self.inventory[title] += quantity

    def remove_book(self, title, quantity):
        if self.inventory[title] < quantity:
            raise False
        self.inventory[title] -= quantity
        if self.inventory[title] == 0:
            del self.inventory[title]

    def view_inventory(self):
        return dict(self.inventory)

    def view_book_quantity(self, title):
        return self.inventory[title]
