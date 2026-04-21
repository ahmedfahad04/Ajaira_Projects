class Order:
    def __init__(self):
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        if any(menu_dish["dish"] == dish["dish"] and menu_dish["count"] >= dish["count"] for menu_dish in self.menu):
            for menu_dish in self.menu:
                if menu_dish["dish"] == dish["dish"]:
                    menu_dish["count"] -= dish["count"]
                    break
            self.selected_dishes.append(dish)
            return True
        return False

    def calculate_total(self):
        return sum(dish["price"] * dish["count"] * self.sales[dish["dish"]] for dish in self.selected_dishes)

    def checkout(self):
        if not self.selected_dishes:
            return False
        total = self.calculate_total()
        self.selected_dishes = []
        return total
