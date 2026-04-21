from functools import reduce
from operator import mul, truediv

class CurrencyConverter:
    def __init__(self):
        self.rates = {
            'USD': 1.0,
            'EUR': 0.85,
            'GBP': 0.72,
            'JPY': 110.15,
            'CAD': 1.23,
            'AUD': 1.34,
            'CNY': 6.40,
        }

    def convert(self, amount, from_currency, to_currency):
        if from_currency == to_currency:
            return amount

        currencies_valid = all(curr in self.rates for curr in [from_currency, to_currency])
        if not currencies_valid:
            return False

        conversion_factors = [amount, truediv(1, self.rates[from_currency]), self.rates[to_currency]]
        return reduce(mul, conversion_factors)

    def get_supported_currencies(self):
        return list(self.rates.keys())

    def add_currency_rate(self, currency, rate):
        exists = currency in self.rates
        if not exists:
            self.rates[currency] = rate
        return not exists

    def update_currency_rate(self, currency, new_rate):
        exists = currency in self.rates
        if exists:
            self.rates[currency] = new_rate
        return exists
