class Order:
    def __init__(self):
        self.menu = []
        self.selected_dishes = []
        self.sales = {}

    def _find_menu_dish(self, dish_name):
        for menu_dish in self.menu:
            if menu_dish["dish"] == dish_name:
                return menu_dish
        return None

    def _update_menu_inventory(self, dish_name, quantity):
        menu_dish = self._find_menu_dish(dish_name)
        if menu_dish and menu_dish["count"] >= quantity:
            menu_dish["count"] -= quantity
            return True
        return menu_dish is None

    def add_dish(self, dish):
        can_add = self._update_menu_inventory(dish["dish"], dish["count"])
        if can_add:
            self.selected_dishes.append(dish)
        return can_add

    def calculate_total(self):
        total = 0
        for dish in self.selected_dishes:
            dish_cost = dish["price"] * dish["count"] * self.sales.get(dish["dish"], 1)
            total += dish_cost
        return total

    def checkout(self):
        empty_cart = len(self.selected_dishes) == 0
        if empty_cart:
            return False
        
        final_amount = self.calculate_total()
        self.selected_dishes = []
        return final_amount
