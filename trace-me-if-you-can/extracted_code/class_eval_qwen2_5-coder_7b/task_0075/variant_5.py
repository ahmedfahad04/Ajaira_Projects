class ShoppingCart:
    def __init__(self):
        self.items = {}

    def manage_item(self, item, price, quantity=1, action='add'):
        if action == 'add':
            if item in self.items:
                self.items[item]['quantity'] += quantity
            else:
                self.items[item] = {'price': price, 'quantity': quantity}
        elif action == 'remove':
            if item in self.items:
                if self.items[item]['quantity'] > quantity:
                    self.items[item]['quantity'] -= quantity
                elif self.items[item]['quantity'] == quantity:
                    del self.items[item]
                else:
                    pass

    def view_items(self) -> dict:
        return self.items

    def total_price(self) -> float:
        return sum([item['quantity'] * item['price'] for item in self.items.values()])
