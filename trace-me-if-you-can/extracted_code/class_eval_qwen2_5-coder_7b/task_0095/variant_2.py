class Store:
    def __init__(self):
        self.items = {}  # Item ID: Item
        self.sales = {}  # Sale ID: Sale

    def add_item(self, item_id, name, quantity):
        if item_id not in self.items:
            self.items[item_id] = {'name': name, 'quantity': quantity}
        else:
            self.items[item_id]['quantity'] += quantity

    def change_item_quantity(self, item_id, quantity):
        if item_id in self.items:
            self.items[item_id]['quantity'] += quantity

    def get_item_quantity(self, item_id):
        return self.items.get(item_id, {}).get('quantity', False)

    def record_sale(self, sale_id, item_id, quantity):
        if self.get_item_quantity(item_id) >= quantity:
            self.change_item_quantity(item_id, -quantity)
            self.sales[sale_id] = {'item_id': item_id, 'quantity': quantity, 'status': 'Finalized'}
        else:
            return False

    def alter_sale_status(self, sale_id, status):
        if sale_id in self.sales:
            self.sales[sale_id]['status'] = status
        else:
            return False

    def check_sale_status(self, sale_id):
        return self.sales.get(sale_id, {}).get('status', False)
