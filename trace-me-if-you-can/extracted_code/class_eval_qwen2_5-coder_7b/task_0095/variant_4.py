class ProductInventory:
    def __init__(self):
        self.products = {}  # Product ID: Product
        self.orders = {}  # Order ID: Order

    def receive_product(self, product_id, name, quantity):
        if product_id not in self.products:
            self.products[product_id] = {'name': name, 'quantity': quantity}
        else:
            self.products[product_id]['quantity'] += quantity

    def modify_product_stock(self, product_id, quantity):
        if product_id in self.products:
            self.products[product_id]['quantity'] += quantity

    def get_product_stock(self, product_id):
        return self.products.get(product_id, {}).get('quantity', False)

    def create_sale(self, sale_id, product_id, quantity):
        if self.get_product_stock(product_id) >= quantity:
            self.modify_product_stock(product_id, -quantity)
            self.orders[sale_id] = {'product_id': product_id, 'quantity': quantity, 'status': 'Confirmed'}
        else:
            return False

    def adjust_sale_status(self, sale_id, status):
        if sale_id in self.orders:
            self.orders[sale_id]['status'] = status
        else:
            return False

    def check_sale_status(self, sale_id):
        return self.orders.get(sale_id, {}).get('status', False)
