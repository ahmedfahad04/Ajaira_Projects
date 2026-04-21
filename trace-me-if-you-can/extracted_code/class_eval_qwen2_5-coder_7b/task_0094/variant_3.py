class CashRegister:
       def __init__(self):
           self.products = {}
           self.customerCredit = 0

       def addProduct(self, itemName, unitPrice, units):
           if not self.updateStock(itemName, units):
               self.products[itemName] = {'unitPrice': unitPrice, 'units': units}

       def acceptPayment(self, amount):
           self.customerCredit += amount
           return self.customerCredit

       def sellItem(self, itemName):
           if itemName in self.products:
               item = self.products[itemName]
               if item['units'] > 0 and self.customerCredit >= item['unitPrice']:
                   self.customerCredit -= item['unitPrice']
                   item['units'] -= 1
                   return self.customerCredit
               else:
                   return False
           else:
               return False

       def updateStock(self, itemName, units):
           if itemName in self.products:
               self.products[itemName]['units'] += units
               return True
           else:
               return False

       def displayInventory(self):
           if not self.products:
               return False
           else:
               inventory = []
               for itemName, productInfo in self.products.items():
                   inventory.append(f"{itemName} - ${productInfo['unitPrice']} [{productInfo['units']}]")
               return "\n".join(inventory)
