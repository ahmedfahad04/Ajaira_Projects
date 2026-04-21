class ShoppingCart:
    def __init__(self):
        self.items = {}

    def add_item(self, item, price, quantity=1):
        item_exists = item in self.items
        self.items[item] = self._build_item_dict(price, quantity)

    def _build_item_dict(self, price, quantity):
        return dict(price=price, quantity=quantity)

    def remove_item(self, item, quantity=1):
        try:
            self.items[item]['quantity'] -= quantity
        except KeyError:
            pass

    def view_items(self) -> dict:
        return dict(iter(self.items.items()))

    def total_price(self) -> float:
        total = 0.0
        for item_data in self.items.values():
            total += item_data['quantity'] * item_data['price']
        return total
