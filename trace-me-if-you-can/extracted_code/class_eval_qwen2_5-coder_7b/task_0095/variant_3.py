class Inventory:
    def __init__(self):
        self.stock = {}  # Product ID: Product
        self.transactions = {}  # Transaction ID: Transaction

    def add_item(self, product_id, name, quantity):
        if product_id not in self.stock:
            self.stock[product_id] = {'name': name, 'quantity': quantity}
        else:
            self.stock[product_id]['quantity'] += quantity

    def update_item_quantity(self, product_id, quantity):
        if product_id in self.stock:
            self.stock[product_id]['quantity'] += quantity

    def get_item_quantity(self, product_id):
        return self.stock.get(product_id, {}).get('quantity', False)

    def process_transaction(self, transaction_id, product_id, quantity):
        if self.get_item_quantity(product_id) >= quantity:
            self.update_item_quantity(product_id, -quantity)
            self.transactions[transaction_id] = {'product_id': product_id, 'quantity': quantity, 'status': 'Completed'}
        else:
            return False

    def change_transaction_status(self, transaction_id, status):
        if transaction_id in self.transactions:
            self.transactions[transaction_id]['status'] = status
        else:
            return False

    def track_transaction(self, transaction_id):
        return self.transactions.get(transaction_id, {}).get('status', False)
