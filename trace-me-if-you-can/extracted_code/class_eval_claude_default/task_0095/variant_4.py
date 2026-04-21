class Warehouse:
    def __init__(self):
        self.inventory = {}
        self.orders = {}

    def add_product(self, product_id, name, quantity):
        product_data = self.inventory.get(product_id, {'name': name, 'quantity': 0})
        product_data['quantity'] += quantity
        self.inventory[product_id] = product_data

    def update_product_quantity(self, product_id, quantity):
        if product_id in self.inventory:
            self.inventory[product_id]['quantity'] = max(0, self.inventory[product_id]['quantity'] + quantity)

    def get_product_quantity(self, product_id):
        product = self.inventory.get(product_id)
        return product['quantity'] if product else False

    def create_order(self, order_id, product_id, quantity):
        current_stock = self.get_product_quantity(product_id)
        if current_stock is False or current_stock < quantity:
            return False
        
        # Process order
        self.inventory[product_id]['quantity'] -= quantity
        self.orders[order_id] = {
            'product_id': product_id, 
            'quantity': quantity, 
            'status': 'Shipped'
        }

    def change_order_status(self, order_id, status):
        if order_id not in self.orders:
            return False
        self.orders[order_id]['status'] = status

    def track_order(self, order_id):
        order = self.orders.get(order_id)
        return order['status'] if order else False
