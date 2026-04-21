class ShoppingCart:
    def __init__(self):
        self.items = {}

    def add_item(self, item, price, quantity=1):
        self.items = {**self.items, item: {'price': price, 'quantity': quantity}}

    def remove_item(self, item, quantity=1):
        if item not in self.items:
            return
        
        updated_quantity = self.items[item]['quantity'] - quantity
        self.items = {**self.items, item: {**self.items[item], 'quantity': updated_quantity}}

    def view_items(self) -> dict:
        return dict(self.items)

    def total_price(self) -> float:
        return sum(details['quantity'] * details['price'] 
                  for details in self.items.values())
