class CheckoutSystem:
    def __init__(self, buyer, products, discount=None):
        self.buyer = buyer
        self.products = products
        self.discount = discount
        self.__base_price = self.get_base_price()

    def get_base_price(self):
        self.__base_price = sum(product['quantity'] * product['price'] for product in self.products)
        return self.__base_price

    def net_amount(self):
        promo_discount = self.discount(self) if self.discount else 0
        return self.__base_price - promo_discount

    @staticmethod
    def frequent_shopper_discount(checkout):
        return checkout.get_base_price() * 0.05 if checkout.buyer['transactions'] >= 100 else 0

    @staticmethod
    def bulk_discount(checkout):
        discount = 0
        for product in checkout.products:
            if product['quantity'] >= 20:
                discount += product['quantity'] * product['price'] * 0.1
        return discount

    @staticmethod
    def diverse_item_discount(checkout):
        return checkout.get_base_price() * 0.07 if len({product['name'] for product in checkout.products}) >= 10 else 0
