class MoneyConverter:
       rates = {
           'USD': 1.0,
           'EUR': 0.85,
           'GBP': 0.72,
           'JPY': 110.15,
           'CAD': 1.23,
           'AUD': 1.34,
           'CNY': 6.40,
       }

       def convert(self, amount, from_curr, to_curr):
           if from_curr == to_curr:
               return amount

           if from_curr not in self.rates or to_curr not in self.rates:
               return False

           from_rate = self.rates[from_curr]
           to_rate = self.rates[to_curr]

           converted_amount = (amount / from_rate) * to_rate
           return converted_amount

       def get_currencies(self):
           return list(self.rates.keys())

       def add_rate(self, currency, rate):
           if currency in self.rates:
               return False
           self.rates[currency] = rate

       def update_rate(self, currency, new_rate):
           if currency not in self.rates:
               return False
           self.rates[currency] = new_rate
