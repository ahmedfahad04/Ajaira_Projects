class Warehouse:
    def __init__(self):
        self.inventory = {}
        self.orders = {}

    def _product_exists(self, product_id):
        return product_id in self.inventory

    def _order_exists(self, order_id):
        return order_id in self.orders

    def add_product(self, product_id, name, quantity):
        self.inventory.setdefault(product_id, {'name': name, 'quantity': 0})
        self.inventory[product_id]['quantity'] += quantity
        if self.inventory[product_id]['name'] == '':
            self.inventory[product_id]['name'] = name

    def update_product_quantity(self, product_id, quantity):
        if self._product_exists(product_id):
            self.inventory[product_id]['quantity'] += quantity

    def get_product_quantity(self, product_id):
        return self.inventory[product_id]['quantity'] if self._product_exists(product_id) else False

    def create_order(self, order_id, product_id, quantity):
        available = self.get_product_quantity(product_id)
        if available is not False and available >= quantity:
            self.update_product_quantity(product_id, -quantity)
            self.orders[order_id] = {'product_id': product_id, 'quantity': quantity, 'status': 'Shipped'}
        else:
            return False

    def change_order_status(self, order_id, status):
        if not self._order_exists(order_id):
            return False
        self.orders[order_id]['status'] = status

    def track_order(self, order_id):
        return self.orders[order_id]['status'] if self._order_exists(order_id) else False
