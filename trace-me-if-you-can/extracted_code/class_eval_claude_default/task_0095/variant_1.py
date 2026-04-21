from collections import defaultdict

class Warehouse:
    def __init__(self):
        self.inventory = defaultdict(lambda: {'name': '', 'quantity': 0})
        self.orders = {}

    def add_product(self, product_id, name, quantity):
        if self.inventory[product_id]['quantity'] == 0:
            self.inventory[product_id] = {'name': name, 'quantity': quantity}
        else:
            self.inventory[product_id]['quantity'] += quantity

    def update_product_quantity(self, product_id, quantity):
        self.inventory[product_id]['quantity'] += quantity

    def get_product_quantity(self, product_id):
        return self.inventory[product_id]['quantity'] if product_id in self.inventory and self.inventory[product_id]['quantity'] > 0 else False

    def create_order(self, order_id, product_id, quantity):
        current_quantity = self.get_product_quantity(product_id)
        if current_quantity and current_quantity >= quantity:
            self.update_product_quantity(product_id, -quantity)
            self.orders[order_id] = {'product_id': product_id, 'quantity': quantity, 'status': 'Shipped'}
            return True
        return False

    def change_order_status(self, order_id, status):
        try:
            self.orders[order_id]['status'] = status
            return True
        except KeyError:
            return False

    def track_order(self, order_id):
        return self.orders.get(order_id, {}).get('status', False)
