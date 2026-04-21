class PricingEngine:
    def __init__(self, user, items, promo=None):
        self.user = user
        self.items = items
        self.promo = promo
        self.__subtotal = self.calculate_subtotal()

    def calculate_subtotal(self):
        self.__subtotal = sum(item['quantity'] * item['price'] for item in self.items)
        return self.__subtotal

    def final_price(self):
        discount = self.promo(self) if self.promo else 0
        return self.__subtotal - discount

    @staticmethod
    def loyalty_discount(pricing):
        return pricing.calculate_subtotal() * 0.05 if pricing.user['points'] >= 1000 else 0

    @staticmethod
    def quantity_discount(pricing):
        discount = 0
        for item in pricing.items:
            if item['quantity'] >= 20:
                discount += item['quantity'] * item['price'] * 0.1
        return discount

    @staticmethod
    def variety_discount(pricing):
        return pricing.calculate_subtotal() * 0.07 if len({item['product'] for item in pricing.items}) >= 10 else 0
