class Order:
    def __init__(self):
        self.menu = {}  # dish_name -> {"price": price, "count": count}
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        dish_name = dish["dish"]
        if dish_name in self.menu:
            if self.menu[dish_name]["count"] < dish["count"]:
                return False
            self.menu[dish_name]["count"] -= dish["count"]
        
        self.selected_dishes.append(dish)
        return True

    def calculate_total(self):
        total_cost = 0
        for selected in self.selected_dishes:
            dish_total = selected["price"] * selected["count"] * self.sales[selected["dish"]]
            total_cost += dish_total
        return total_cost

    def checkout(self):
        has_items = len(self.selected_dishes) > 0
        if has_items:
            final_total = self.calculate_total()
            self.selected_dishes = []
            return final_total
        return False
