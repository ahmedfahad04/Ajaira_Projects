class LibraryCatalog:
        def __init__(self):
            self.stock = {}

        def add_title(self, title, quantity=1):
            if title in self.stock:
                self.stock[title] += quantity
            else:
                self.stock[title] = quantity

        def subtract_title(self, title, quantity):
            if title not in self.stock or self.stock[title] < quantity:
                raise ValueError("Insufficient quantity in inventory")
            self.stock[title] -= quantity
            if self.stock[title] == 0:
                del self.stock[title]

        def display_stock(self):
            return self.stock

        def check_title_quantity(self, title):
            return self.stock.get(title, 0)
