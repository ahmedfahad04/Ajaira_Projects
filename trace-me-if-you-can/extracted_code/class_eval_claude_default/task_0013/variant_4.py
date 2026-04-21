class BookManagement:
    def __init__(self):
        self.inventory = {}

    def add_book(self, title, quantity=1):
        self.inventory.setdefault(title, 0)
        self.inventory[title] += quantity

    def remove_book(self, title, quantity):
        current_stock = self.inventory.setdefault(title, 0)
        if current_stock < quantity:
            raise False
        self.inventory[title] = current_stock - quantity
        if self.inventory[title] == 0:
            self.inventory.pop(title)

    def view_inventory(self):
        return self.inventory

    def view_book_quantity(self, title):
        return self.inventory.setdefault(title, 0)
