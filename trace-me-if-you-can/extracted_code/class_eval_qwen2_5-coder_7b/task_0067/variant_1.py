class Order:
    def __init__(self):
        self.menu = {}
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        dish_name = dish["dish"]
        count = dish["count"]
        if dish_name in self.menu:
            if self.menu[dish_name]["count"] < count:
                return False
            self.menu[dish_name]["count"] -= count
        else:
            self.menu[dish_name] = {"price": dish["price"], "count": dish["count"]}
        self.selected_dishes.append(dish)
        return True

    def calculate_total(self):
        total = 0
        for dish in self.selected_dishes:
            total += dish["price"] * dish["count"] * self.sales[dish["dish"]]
        return total

    def checkout(self):
        if not self.selected_dishes:
            return False
        total = self.calculate_total()
        self.selected_dishes = []
        return total
