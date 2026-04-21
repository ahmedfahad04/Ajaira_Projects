class OrderItem:
    def __init__(self, dish, price, count):
        self.dish = dish
        self.price = price
        self.count = count

class Order:
    def __init__(self):
        self.menu = []
        self.items = []
        self.sales = {}

    def add_dish(self, dish):
        for menu_dish in self.menu:
            if menu_dish["dish"] == dish["dish"]:
                if menu_dish["count"] < dish["count"]:
                    return False
                menu_dish["count"] -= dish["count"]
                break
        else:
            self.menu.append(dish)
        self.items.append(dish)
        return True

    def calculate_total(self):
        total = 0
        for item in self.items:
            total += item["price"] * item["count"] * self.sales[item["dish"]]
        return total

    def checkout(self):
        if not self.items:
            return False
        total = self.calculate_total()
        self.items = []
        return total
