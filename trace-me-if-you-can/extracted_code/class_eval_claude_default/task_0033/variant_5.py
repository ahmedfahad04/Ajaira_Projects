class DiscountStrategy:
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.__total = None

    def total(self):
        if self.__total is None:
            self.__total = self._calculate_cart_total()
        return self.__total

    def _calculate_cart_total(self):
        return sum(item['quantity'] * item['price'] for item in self.cart)

    def due(self):
        total_amount = self.total()
        discount_amount = self._apply_promotion() if self.promotion else 0
        return total_amount - discount_amount

    def _apply_promotion(self):
        return self.promotion(self)

    @staticmethod
    def FidelityPromo(order):
        customer_fidelity = order.customer['fidelity']
        return order.total() * 0.05 if customer_fidelity >= 1000 else 0

    @staticmethod
    def BulkItemPromo(order):
        total_discount = 0
        for item in order.cart:
            item_quantity = item['quantity']
            if item_quantity >= 20:
                total_discount += item_quantity * item['price'] * 0.1
        return total_discount

    @staticmethod
    def LargeOrderPromo(order):
        unique_products = len(set(item['product'] for item in order.cart))
        return order.total() * 0.07 if unique_products >= 10 else 0
