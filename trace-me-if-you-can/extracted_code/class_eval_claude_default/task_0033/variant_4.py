class DiscountStrategy:
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.__total = self.total()

    def _item_totals(self):
        for item in self.cart:
            yield item['quantity'] * item['price']

    def total(self):
        self.__total = sum(self._item_totals())
        return self.__total

    def due(self):
        return self.__total - (self.promotion(self) if self.promotion else 0)

    @staticmethod
    def FidelityPromo(order):
        return order.total() * 0.05 if order.customer.get('fidelity', 0) >= 1000 else 0

    @staticmethod
    def BulkItemPromo(order):
        def bulk_discounts():
            for item in order.cart:
                if item['quantity'] >= 20:
                    yield item['quantity'] * item['price'] * 0.1
        return sum(bulk_discounts())

    @staticmethod
    def LargeOrderPromo(order):
        product_set = set(item['product'] for item in order.cart)
        return order.total() * 0.07 if len(product_set) >= 10 else 0
