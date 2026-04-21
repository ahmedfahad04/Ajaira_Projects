from collections import defaultdict

class Order:
    def __init__(self):
        self.menu = []
        self.selected_dishes = []
        self.sales = defaultdict(lambda: 1)

    def add_dish(self, dish):
        dish_name = dish["dish"]
        required_count = dish["count"]
        
        for i, menu_item in enumerate(self.menu):
            if menu_item["dish"] == dish_name:
                available_count = menu_item["count"]
                if available_count >= required_count:
                    self.menu[i]["count"] = available_count - required_count
                    self.selected_dishes.append(dish)
                    return True
                return False
        
        self.selected_dishes.append(dish)
        return True

    def calculate_total(self):
        def get_dish_cost(dish):
            return dish["price"] * dish["count"] * self.sales[dish["dish"]]
        
        return sum(map(get_dish_cost, self.selected_dishes))

    def checkout(self):
        if self.selected_dishes:
            total = self.calculate_total()
            self.selected_dishes = []
            return total
        return False
