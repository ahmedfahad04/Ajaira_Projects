class CurrencyExchange:
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

       def exchange(self, amount, source_currency, target_currency):
           if source_currency == target_currency:
               return amount

           if source_currency not in self.currency_rates or target_currency not in self.currency_rates:
               return False

           source_rate = self.currency_rates[source_currency]
           target_rate = self.currency_rates[target_currency]

           converted_amount = (amount / source_rate) * target_rate
           return converted_amount

       def list_supported_currencies(self):
           return list(self.currency_rates.keys())

       def introduce_currency_rate(self, currency, rate):
           if currency in self.currency_rates:
               return False
           self.currency_rates[currency] = rate

       def refresh_currency_rate(self, currency, new_rate):
           if currency not in self.currency_rates:
               return False
           self.currency_rates[currency] = new_rate
