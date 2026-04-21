class MenuItem:
    def __init__(self, dish, price, count):
        self.dish = dish
        self.price = price
        self.count = count

class Order:
    def __init__(self):
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        for menu_dish in self.menu:
            if menu_dish.dish == dish["dish"]:
                if menu_dish.count < dish["count"]:
                    return False
                menu_dish.count -= dish["count"]
                break
        else:
            new_item = MenuItem(dish["dish"], dish["price"], dish["count"])
            self.menu.append(new_item)
        self.selected_dishes.append(dish)
        return True

    def calculate_total(self):
        total = 0
        for dish in self.selected_dishes:
            for menu_item in self.menu:
                if menu_item.dish == dish["dish"]:
                    total += dish["price"] * dish["count"] * self.sales[dish["dish"]]
                    break
        return total

    def checkout(self):
        if not self.selected_dishes:
            return False
        total = self.calculate_total()
        self.selected_dishes = []
        return total
