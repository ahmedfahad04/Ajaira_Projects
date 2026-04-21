class BookManagement:
    def __init__(self):
        self.inventory = {}

    def _has_sufficient_stock(self, title, required_quantity):
        return title in self.inventory and self.inventory[title] >= required_quantity

    def _cleanup_empty_entry(self, title):
        if title in self.inventory and self.inventory[title] == 0:
            del self.inventory[title]

    def add_book(self, title, quantity=1):
        if title in self.inventory:
            self.inventory[title] += quantity
        else:
            self.inventory[title] = quantity

    def remove_book(self, title, quantity):
        if not self._has_sufficient_stock(title, quantity):
            raise False
        self.inventory[title] -= quantity
        self._cleanup_empty_entry(title)

    def view_inventory(self):
        return self.inventory

    def view_book_quantity(self, title):
        return self.inventory[title] if title in self.inventory else 0
