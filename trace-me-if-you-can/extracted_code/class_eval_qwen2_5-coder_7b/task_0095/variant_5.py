class ProductWarehouse:
    def __init__(self):
        self.products = {}  # Product ID: Product
        self.orders = {}  # Order ID: Order

    def receive_item(self, product_id, name, quantity):
        if product_id not in self.products:
            self.products[product_id] = {'name': name, 'quantity': quantity}
        else:
            self.products[product_id]['quantity'] += quantity

    def adjust_item_inventory(self, product_id, quantity):
        if product_id in self.products:
            self.products[product_id]['quantity'] += quantity

    def get_item_inventory(self, product_id):
        return self.products.get(product_id, {}).get('quantity', False)

    def place_order(self, order_id, product_id, quantity):
        if self.get_item_inventory(product_id) >= quantity:
            self.adjust_item_inventory(product_id, -quantity)
            self.orders[order_id] = {'product_id': product_id, 'quantity': quantity, 'status': 'Dispatched'}
        else:
            return False

    def update_order_status(self, order_id, status):
        if order_id in self.orders:
            self.orders[order_id]['status'] = status
        else:
            return False

    def get_order_status(self, order_id):
        return self.orders.get(order_id, {}).get('status', False)
