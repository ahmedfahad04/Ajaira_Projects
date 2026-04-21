class Product:
    def __init__(self, name, quantity):
        self.name = name
        self.quantity = quantity

class Order:
    def __init__(self, product_id, quantity, status='Shipped'):
        self.product_id = product_id
        self.quantity = quantity
        self.status = status

class Warehouse:
    def __init__(self):
        self.inventory = {}
        self.orders = {}

    def add_product(self, product_id, name, quantity):
        if product_id in self.inventory:
            self.inventory[product_id].quantity += quantity
        else:
            self.inventory[product_id] = Product(name, quantity)

    def update_product_quantity(self, product_id, quantity):
        if product_id in self.inventory:
            self.inventory[product_id].quantity += quantity

    def get_product_quantity(self, product_id):
        product = self.inventory.get(product_id)
        return product.quantity if product else False

    def create_order(self, order_id, product_id, quantity):
        available_quantity = self.get_product_quantity(product_id)
        if available_quantity and available_quantity >= quantity:
            self.update_product_quantity(product_id, -quantity)
            self.orders[order_id] = Order(product_id, quantity)
            return True
        return False

    def change_order_status(self, order_id, status):
        order = self.orders.get(order_id)
        if order:
            order.status = status
            return True
        return False

    def track_order(self, order_id):
        order = self.orders.get(order_id)
        return order.status if order else False
