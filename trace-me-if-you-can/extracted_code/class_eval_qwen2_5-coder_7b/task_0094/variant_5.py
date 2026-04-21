class CoinSlot:
       def __init__(self):
           self.items = {}
           self.funds = 0

       def addProduct(self, name, price, quantity):
           if not self.updateStock(name, quantity):
               self.items[name] = {'price': price, 'quantity': quantity}

       def insertCoin(self, amount):
           self.funds += amount
           return self.funds

       def sellItem(self, name):
           if name in self.items:
               product = self.items[name]
               if product['quantity'] > 0 and self.funds >= product['price']:
                   self.funds -= product['price']
                   product['quantity'] -= 1
                   return self.funds
               else:
                   return False
           else:
               return False

       def updateStock(self, name, quantity):
           if name in self.items:
               self.items[name]['quantity'] += quantity
               return True
           else:
               return False

       def listProducts(self):
           if not self.items:
               return False
           else:
               products = []
               for name, details in self.items.items():
                   products.append(f"{name} - ${details['price']} [{details['quantity']}]")
               return "\n".join(products)
