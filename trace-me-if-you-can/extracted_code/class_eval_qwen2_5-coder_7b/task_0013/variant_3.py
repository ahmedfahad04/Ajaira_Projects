class Bookkeeping:
        def __init__(self):
            self.items = {}

        def insert_book(self, title, quantity=1):
            if title in self.items:
                self.items[title] += quantity
            else:
                self.items[title] = quantity

        def decrement_book(self, title, quantity):
            if title not in self.items or self.items[title] < quantity:
                raise IndexError("Book quantity insufficient or not available")
            self.items[title] -= quantity
            if self.items[title] == 0:
                del self.items[title]

        def list_inventory(self):
            return self.items

        def count_books(self, title):
            return self.items.get(title, 0)
