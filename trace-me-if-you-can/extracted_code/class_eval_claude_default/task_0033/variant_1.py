def calculate_total(cart):
    return sum(item['quantity'] * item['price'] for item in cart)

def fidelity_discount(customer, total):
    return total * 0.05 if customer['fidelity'] >= 1000 else 0

def bulk_item_discount(cart):
    discount = 0
    for item in cart:
        if item['quantity'] >= 20:
            discount += item['quantity'] * item['price'] * 0.1
    return discount

def large_order_discount(cart, total):
    return total * 0.07 if len({item['product'] for item in cart}) >= 10 else 0

class DiscountStrategy:
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.__total = self.total()

    def total(self):
        self.__total = calculate_total(self.cart)
        return self.__total

    def due(self):
        discount = self.promotion(self) if self.promotion else 0
        return self.__total - discount

    @staticmethod
    def FidelityPromo(order):
        return fidelity_discount(order.customer, order.total())

    @staticmethod
    def BulkItemPromo(order):
        return bulk_item_discount(order.cart)

    @staticmethod
    def LargeOrderPromo(order):
        return large_order_discount(order.cart, order.total())
