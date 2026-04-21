class Shop:
       def __init__(self):
           self.stock = {}
           self.credit = 0

       def addProduct(self, productName, cost, count):
           if not self.refillStock(productName, count):
               self.stock[productName] = {'cost': cost, 'count': count}

       def loadCoin(self, value):
           self.credit += value
           return self.credit

       def buyProduct(self, productName):
           if productName in self.stock:
               item = self.stock[productName]
               if item['count'] > 0 and self.credit >= item['cost']:
                   self.credit -= item['cost']
                   item['count'] -= 1
                   return self.credit
               else:
                   return False
           else:
               return False

       def refillStock(self, productName, count):
           if productName in self.stock:
               self.stock[productName]['count'] += count
               return True
           else:
               return False

       def showProducts(self):
           if not self.stock:
               return False
           else:
               products = []
               for productName, productInfo in self.stock.items():
                   products.append(f"{productName} - ${productInfo['cost']} [{productInfo['count']}]")
               return "\n".join(products)
