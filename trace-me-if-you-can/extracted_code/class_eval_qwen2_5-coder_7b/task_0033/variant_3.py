class ShoppingCart:
    def __init__(self, buyer, items, promotion=None):
        self.buyer = buyer
        self.items = items
        self.promotion = promotion
        self.__subtotal = self.compute_subtotal()

    def compute_subtotal(self):
        self.__subtotal = sum(item['quantity'] * item['price'] for item in self.items)
        return self.__subtotal

    def compute_final_price(self):
        promo_discount = self.promotion(self) if self.promotion else 0
        return self.__subtotal - promo_discount

    @staticmethod
    def high_fidelity_discount(cart):
        return cart.compute_subtotal() * 0.05 if cart.buyer['years'] >= 5 else 0

    @staticmethod
    def large_quantity_discount(cart):
        discount = 0
        for item in cart.items:
            if item['quantity'] >= 20:
                discount += item['quantity'] * item['price'] * 0.1
        return discount

    @staticmethod
    def broad_item_discount(cart):
        return cart.compute_subtotal() * 0.07 if len({item['category'] for item in cart.items}) >= 5 else 0
