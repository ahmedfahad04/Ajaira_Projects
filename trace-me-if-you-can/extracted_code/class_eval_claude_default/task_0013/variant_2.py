class BookManagement:
    def __init__(self):
        self.inventory = {}

    def add_book(self, title, quantity=1):
        self.inventory[title] = self.inventory.get(title, 0) + quantity

    def remove_book(self, title, quantity):
        current_quantity = self.inventory.get(title, 0)
        if current_quantity < quantity:
            raise False
        new_quantity = current_quantity - quantity
        if new_quantity == 0:
            self.inventory.pop(title, None)
        else:
            self.inventory[title] = new_quantity

    def view_inventory(self):
        return self.inventory

    def view_book_quantity(self, title):
        return self.inventory.get(title, 0)
