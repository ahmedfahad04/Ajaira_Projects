class Cart:
    def __init__(self):
        self.products = {}

    def insert_product(self, product, cost, units=1):
        if product in self.products:
            self.products[product] = {'cost': cost, 'units': units}
        else:
            self.products[product] = {'cost': cost, 'units': units}

    def delete_product(self, product, units=1):
        if product in self.products:
            self.products[product]['units'] -= units
            if self.products[product]['units'] <= 0:
                del self.products[product]

    def display_contents(self) -> dict:
        return self.products

    def calculate_total(self) -> float:
        return sum([product['units'] * product['cost'] for product in self.products.values()])
