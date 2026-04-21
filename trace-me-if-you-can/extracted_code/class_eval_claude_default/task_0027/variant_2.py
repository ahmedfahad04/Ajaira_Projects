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

    def _is_valid_currency(self, currency):
        return currency in self.rates

    def _convert_to_usd(self, amount, currency):
        return amount / self.rates[currency]

    def _convert_from_usd(self, usd_amount, currency):
        return usd_amount * self.rates[currency]

    def convert(self, amount, from_currency, to_currency):
        if from_currency == to_currency:
            return amount

        if not (self._is_valid_currency(from_currency) and self._is_valid_currency(to_currency)):
            return False

        usd_amount = self._convert_to_usd(amount, from_currency)
        return self._convert_from_usd(usd_amount, to_currency)

    def get_supported_currencies(self):
        return list(self.rates.keys())

    def add_currency_rate(self, currency, rate):
        if currency in self.rates:
            return False
        self.rates[currency] = rate

    def update_currency_rate(self, currency, new_rate):
        if currency not in self.rates:
            return False
        self.rates[currency] = new_rate
