class Order:
    def __init__(self):
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        menu_item = next((item for item in self.menu if item["dish"] == dish["dish"]), None)
        if menu_item and menu_item["count"] >= dish["count"]:
            menu_item["count"] -= dish["count"]
            self.selected_dishes.append(dish)
            return True
        elif menu_item:
            return False
        else:
            self.selected_dishes.append(dish)
            return True

    def calculate_total(self):
        return sum(dish["price"] * dish["count"] * self.sales[dish["dish"]] 
                  for dish in self.selected_dishes)

    def checkout(self):
        if not self.selected_dishes:
            return False
        total = self.calculate_total()
        self.selected_dishes.clear()
        return total
