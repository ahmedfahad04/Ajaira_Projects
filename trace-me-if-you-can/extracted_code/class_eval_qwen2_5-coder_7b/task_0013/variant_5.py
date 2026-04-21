class InventoryManager:
        def __init__(self):
            self.collection = {}

        def add_entry(self, title, quantity=1):
            if title in self.collection:
                self.collection[title] += quantity
            else:
                self.collection[title] = quantity

        def remove_entry(self, title, quantity):
            if title not in self.collection or self.collection[title] < quantity:
                raise LookupError("Book not in inventory or insufficient quantity")
            self.collection[title] -= quantity
            if self.collection[title] == 0:
                del self.collection[title]

        def display_collection(self):
            return self.collection

        def check_quantity(self, title):
            return self.collection.get(title, 0)
