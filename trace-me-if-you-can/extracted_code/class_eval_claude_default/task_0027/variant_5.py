class CurrencyConverter:
    _DEFAULT_RATES = {
        'USD': 1.0,
        'EUR': 0.85,
        'GBP': 0.72,
        'JPY': 110.15,
        'CAD': 1.23,
        'AUD': 1.34,
        'CNY': 6.40,
    }

    def __init__(self):
        self.rates = self._DEFAULT_RATES.copy()

    def convert(self, amount, from_currency, to_currency):
        conversion_pipeline = [
            lambda: amount if from_currency == to_currency else None,
            lambda: self._perform_conversion(amount, from_currency, to_currency) 
                   if self._validate_currencies(from_currency, to_currency) else False
        ]
        
        for step in conversion_pipeline:
            result = step()
            if result is not None:
                return result
        
        return False

    def _validate_currencies(self, from_curr, to_curr):
        return from_curr in self.rates and to_curr in self.rates

    def _perform_conversion(self, amount, from_curr, to_curr):
        return (amount / self.rates[from_curr]) * self.rates[to_curr]

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
