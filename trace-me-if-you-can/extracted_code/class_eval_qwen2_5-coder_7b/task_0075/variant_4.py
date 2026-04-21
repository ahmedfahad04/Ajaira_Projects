class ShoppingList:
    def __init__(self):
        self.items = {}

    def add(self, item, price, quantity=1):
        if item in self.items:
            self.items[item]['quantity'] += quantity
        else:
            self.items[item] = {'price': price, 'quantity': quantity}

    def subtract(self, item, quantity=1):
        if item in self.items:
            self.items[item]['quantity'] -= quantity
            if self.items[item]['quantity'] < 0:
                self.items[item]['quantity'] = 0

    def display(self) -> dict:
        return self.items

    def total(self) -> float:
        return sum([self.items[item]['quantity'] * self.items[item]['price'] for item in self.items])
