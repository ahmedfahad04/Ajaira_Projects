class BookManagement:
    def __init__(self):
        self.inventory = {}

    def add_book(self, title, quantity=1):
        try:
            self.inventory[title] += quantity
        except KeyError:
            self.inventory[title] = quantity

    def remove_book(self, title, quantity):
        try:
            if self.inventory[title] < quantity:
                raise False
            self.inventory[title] -= quantity
            if self.inventory[title] == 0:
                del self.inventory[title]
        except KeyError:
            raise False

    def view_inventory(self):
        return self.inventory

    def view_book_quantity(self, title):
        try:
            return self.inventory[title]
        except KeyError:
            return 0
