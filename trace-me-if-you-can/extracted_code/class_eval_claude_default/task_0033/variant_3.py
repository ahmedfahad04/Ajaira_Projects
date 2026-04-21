class DiscountStrategy:
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self._total_cache = None

    @property
    def _total(self):
        if self._total_cache is None:
            self._total_cache = self.total()
        return self._total_cache

    def total(self):
        return sum(item['quantity'] * item['price'] for item in self.cart)

    def due(self):
        discount = self.promotion(self) if self.promotion is not None else 0
        return self._total - discount

    @staticmethod
    def FidelityPromo(order):
        has_fidelity = order.customer['fidelity'] >= 1000
        return order._total * 0.05 if has_fidelity else 0

    @staticmethod
    def BulkItemPromo(order):
        discount = 0
        for item in order.cart:
            if item['quantity'] >= 20:
                discount += item['quantity'] * item['price'] * 0.1
        return discount

    @staticmethod
    def LargeOrderPromo(order):
        distinct_products = {item['product'] for item in order.cart}
        return order._total * 0.07 if len(distinct_products) >= 10 else 0
