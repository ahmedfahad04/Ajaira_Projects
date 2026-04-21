class ShoppingCart:
    def __init__(self):
        self._items = {}

    def _create_item_entry(self, price, quantity):
        return {'price': price, 'quantity': quantity}

    def add_item(self, item, price, quantity=1):
        self._items[item] = self._create_item_entry(price, quantity)

    def remove_item(self, item, quantity=1):
        item_data = self._items.get(item)
        if item_data is not None:
            item_data['quantity'] -= quantity

    @property
    def items(self) -> dict:
        return self._items.copy()

    def view_items(self) -> dict:
        return self.items

    def total_price(self) -> float:
        return sum(map(lambda x: x['quantity'] * x['price'], self._items.values()))
