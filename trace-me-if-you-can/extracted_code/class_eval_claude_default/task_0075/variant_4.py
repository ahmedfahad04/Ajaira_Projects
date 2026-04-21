class ShoppingCart:
    def __init__(self):
        self.items = {}

    def add_item(self, item, price, quantity=1):
        self.items.setdefault(item, {})['price'] = price
        self.items[item]['quantity'] = quantity

    def remove_item(self, item, quantity=1):
        current_item = self.items.get(item, {})
        if current_item:
            current_item['quantity'] = current_item.get('quantity', 0) - quantity

    def view_items(self) -> dict:
        return {k: v for k, v in self.items.items()}

    def total_price(self) -> float:
        calculation = [item_info['quantity'] * item_info['price'] 
                      for item_info in self.items.values()]
        return sum(calculation)
