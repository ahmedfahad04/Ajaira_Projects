class ExchangeService:
       def __init__(self):
           self.currency_rates = {
               'USD': 1.0,
               'EUR': 0.85,
               'GBP': 0.72,
               'JPY': 110.15,
               'CAD': 1.23,
               'AUD': 1.34,
               'CNY': 6.40,
           }

       def calculate_conversion(self, amount, from_currency, to_currency):
           if from_currency == to_currency:
               return amount

           if from_currency not in self.currency_rates or to_currency not in self.currency_rates:
               return False

           from_rate = self.currency_rates[from_currency]
           to_rate = self.currency_rates[to_currency]

           converted_amount = (amount / from_rate) * to_rate
           return converted_amount

       def retrieve_supported_currencies(self):
           return list(self.currency_rates.keys())

       def add_exchange_rate(self, currency, rate):
           if currency in self.currency_rates:
               return False
           self.currency_rates[currency] = rate

       def update_exchange_rate(self, currency, new_rate):
           if currency not in self.currency_rates:
               return False
           self.currency_rates[currency] = new_rate
