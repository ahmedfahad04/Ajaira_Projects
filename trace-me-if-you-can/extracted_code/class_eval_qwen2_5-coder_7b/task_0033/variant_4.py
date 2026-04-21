class OrderProcessor:
    def __init__(self, customer, items, promo=None):
        self.customer = customer
        self.items = items
        self.promo = promo
        self.__price_before_promo = self.compute_price_before_promo()

    def compute_price_before_promo(self):
        self.__price_before_promo = sum(item['quantity'] * item['price'] for item in self.items)
        return self.__price_before_promo

    def calculate_final_price(self):
        promo_discount = self.promo(self) if self.promo else 0
        return self.__price_before_promo - promo_discount

    @staticmethod
    def gold_member_discount(order):
        return order.calculate_price_before_promo() * 0.05 if order.customer['status'] == 'gold' else 0

    @staticmethod
    def bulk_purchase_discount(order):
        discount = 0
        for item in order.items:
            if item['quantity'] >= 20:
                discount += item['quantity'] * item['price'] * 0.1
        return discount

    @staticmethod
    def variety_discount(order):
        return order.calculate_price_before_promo() * 0.07 if len({item['item_type'] for item in order.items}) >= 10 else 0
