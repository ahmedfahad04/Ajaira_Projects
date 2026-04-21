class StoreCheckout:
    def __init__(self, customer, products, discount=None):
        self.customer = customer
        self.products = products
        self.discount = discount
        self.__base_cost = self.calculate_base_cost()

    def calculate_base_cost(self):
        self.__base_cost = sum(product['quantity'] * product['price'] for product in self.products)
        return self.__base_cost

    def compute_net_price(self):
        promo_discount = self.discount(self) if self.discount else 0
        return self.__base_cost - promo_discount

    @staticmethod
    def long_term_customer_discount(checkout):
        return checkout.calculate_base_cost() * 0.05 if checkout.customer['years'] >= 3 else 0

    @staticmethod
    def high_quantity_discount(checkout):
        discount = 0
        for product in checkout.products:
            if product['quantity'] >= 20:
                discount += product['quantity'] * product['price'] * 0.1
        return discount

    @staticmethod
    def diverse_product_discount(checkout):
        return checkout.calculate_base_cost() * 0.07 if len({product['category'] for product in checkout.products}) >= 5 else 0
