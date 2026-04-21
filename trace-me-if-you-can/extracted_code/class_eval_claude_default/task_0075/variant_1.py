from collections import defaultdict
from dataclasses import dataclass

@dataclass
class CartItem:
    price: float
    quantity: int = 1

class ShoppingCart:
    def __init__(self):
        self.items = defaultdict(lambda: CartItem(0, 0))

    def add_item(self, item, price, quantity=1):
        self.items[item] = CartItem(price, quantity)

    def remove_item(self, item, quantity=1):
        if item in self.items:
            self.items[item].quantity -= quantity

    def view_items(self) -> dict:
        return {k: {'price': v.price, 'quantity': v.quantity} 
                for k, v in self.items.items() if v.quantity > 0}

    def total_price(self) -> float:
        return sum(item.quantity * item.price for item in self.items.values())
