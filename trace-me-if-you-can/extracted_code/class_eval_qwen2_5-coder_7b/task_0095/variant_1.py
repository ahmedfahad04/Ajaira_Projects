class InventoryManager:
    def __init__(self):
        self.products = {}  # Product ID: Product
        self.sales = {}  # Sale ID: Sale

    def store_product(self, product_id, name, quantity):
        if product_id not in self.products:
            self.products[product_id] = {'name': name, 'quantity': quantity}
        else:
            self.products[product_id]['quantity'] += quantity

    def adjust_product_quantity(self, product_id, quantity):
        if product_id in self.products:
            self.products[product_id]['quantity'] += quantity

    def fetch_product_quantity(self, product_id):
        return self.products.get(product_id, {}).get('quantity', False)

    def process_sale(self, sale_id, product_id, quantity):
        if self.fetch_product_quantity(product_id) >= quantity:
            self.adjust_product_quantity(product_id, -quantity)
            self.sales[sale_id] = {'product_id': product_id, 'quantity': quantity, 'status': 'Processed'}
        else:
            return False

    def modify_sale_status(self, sale_id, status):
        if sale_id in self.sales:
            self.sales[sale_id]['status'] = status
        else:
            return False

    def monitor_sale(self, sale_id):
        return self.sales.get(sale_id, {}).get('status', False)
