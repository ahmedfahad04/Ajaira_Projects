class ShoppingCart:
    def __init__(self):
        self.products = {}

    def include_item(self, product, price, amount=1):
        if product in self.products:
            self.products[product]['amount'] += amount
        else:
            self.products[product] = {'price': price, 'amount': amount}

    def exclude_item(self, product, amount=1):
        if product in self.products:
            self.products[product]['amount'] -= amount
            if self.products[product]['amount'] < 0:
                self.products[product]['amount'] = 0

    def show_items(self) -> dict:
        return self.products

    def compute_total(self) -> float:
        return sum([self.products[product]['amount'] * self.products[product]['price'] for product in self.products])
