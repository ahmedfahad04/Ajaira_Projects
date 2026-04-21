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
        conversion_map = {
            (from_currency, to_currency): amount 
            for from_currency in self.rates 
            for to_currency in self.rates
        }
        
        if (from_currency, to_currency) not in conversion_map:
            return False
        
        if from_currency == to_currency:
            return amount
            
        return (amount / self.rates[from_currency]) * self.rates[to_currency]

    def get_supported_currencies(self):
        return list(self.rates.keys())

    def add_currency_rate(self, currency, rate):
        if self.rates.get(currency) is not None:
            return False
        self.rates[currency] = rate

    def update_currency_rate(self, currency, new_rate):
        if self.rates.get(currency) is None:
            return False
        self.rates[currency] = new_rate
