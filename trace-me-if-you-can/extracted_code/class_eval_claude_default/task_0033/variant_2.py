class DiscountStrategy:
    PROMOTIONS = {}
    
    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
        self.__total = self.total()

    def total(self):
        self.__total = sum(item['quantity'] * item['price'] for item in self.cart)
        return self.__total

    def due(self):
        discount = 0 if self.promotion is None else self.promotion(self)
        return self.__total - discount

    @classmethod
    def register_promotion(cls, name, func):
        cls.PROMOTIONS[name] = func
        return func

    @staticmethod
    def FidelityPromo(order):
        return order.total() * 0.05 if order.customer['fidelity'] >= 1000 else 0

    @staticmethod  
    def BulkItemPromo(order):
        return sum(item['quantity'] * item['price'] * 0.1 
                  for item in order.cart if item['quantity'] >= 20)

    @staticmethod
    def LargeOrderPromo(order):
        unique_products = len({item['product'] for item in order.cart})
        return order.total() * 0.07 if unique_products >= 10 else 0
