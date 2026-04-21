class Basket:
    def __init__(self):
        self.goods = {}

    def add_product(self, item, price, count=1):
        if item in self.goods:
            self.goods[item]['count'] += count
        else:
            self.goods[item] = {'price': price, 'count': count}

    def subtract_product(self, item, count=1):
        if item in self.goods:
            self.goods[item]['count'] -= count
            if self.goods[item]['count'] == 0:
                del self.goods[item]

    def list_contents(self) -> dict:
        return self.goods

    def get_total_cost(self) -> float:
        return sum([self.goods[item]['count'] * self.goods[item]['price'] for item in self.goods])
