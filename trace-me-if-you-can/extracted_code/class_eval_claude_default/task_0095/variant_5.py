class Warehouse:
    SHIPPED_STATUS = 'Shipped'
    
    def __init__(self):
        self.inventory = {}
        self.orders = {}

    def add_product(self, product_id, name, quantity):
        if product_id not in self.inventory:
            self.inventory[product_id] = {'name': name, 'quantity': quantity}
            return
        self.inventory[product_id]['quantity'] += quantity

    def update_product_quantity(self, product_id, quantity):
        product = self.inventory.get(product_id)
        if product is not None:
            product['quantity'] += quantity

    def get_product_quantity(self, product_id):
        product = self.inventory.get(product_id)
        return product['quantity'] if product is not None else False

    def create_order(self, order_id, product_id, quantity):
        available_stock = self.get_product_quantity(product_id)
        
        # Check stock availability
        if available_stock is False or quantity > available_stock:
            return False
            
        # Fulfill order
        self.update_product_quantity(product_id, -quantity)
        self.orders[order_id] = {
            'product_id': product_id,
            'quantity': quantity,
            'status': self.SHIPPED_STATUS
        }

    def change_order_status(self, order_id, status):
        order = self.orders.get(order_id)
        if order is None:
            return False
        order['status'] = status

    def track_order(self, order_id):
        order = self.orders.get(order_id)
        return order['status'] if order is not None else False
