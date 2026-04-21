class Order:
    def __init__(self):
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def add_dish(self, dish):
        target_dish = dish["dish"]
        requested_count = dish["count"]
        
        menu_matches = [m for m in self.menu if m["dish"] == target_dish]
        
        if menu_matches:
            menu_dish = menu_matches[0]
            if menu_dish["count"] < requested_count:
                return False
            menu_dish["count"] -= requested_count
        
        self.selected_dishes.append(dish)
        return True

    def calculate_total(self):
        def calculate_dish_subtotal(dish):
            base_cost = dish["price"] * dish["count"]
            multiplier = self.sales[dish["dish"]]
            return base_cost * multiplier
        
        subtotals = [calculate_dish_subtotal(dish) for dish in self.selected_dishes]
        return sum(subtotals)

    def checkout(self):
        if not self.selected_dishes:
            return False
        
        order_total = self.calculate_total()
        self.selected_dishes = []
        return order_total
