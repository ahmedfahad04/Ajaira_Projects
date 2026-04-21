class Machine:
       def __init__(self):
           self.store = {}
           self.cash = 0

       def addItem(self, item, price, quantity):
           if not self.replenishItem(item, quantity):
               self.store[item] = {'price': price, 'quantity': quantity}

       def depositCoin(self, amount):
           self.cash += amount
           return self.cash

       def purchase(self, item):
           if item in self.store:
               item_details = self.store[item]
               if item_details['quantity'] > 0 and self.cash >= item_details['price']:
                   self.cash -= item_details['price']
                   item_details['quantity'] -= 1
                   return self.cash
               else:
                   return False
           else:
               return False

       def replenishItem(self, item, quantity):
           if item in self.store:
               self.store[item]['quantity'] += quantity
               return True
           else:
               return False

       def displayItems(self):
           if not self.store:
               return False
           else:
               items = []
               for item, details in self.store.items():
                   items.append(f"{item} - ${details['price']} [{details['quantity']}]")
               return "\n".join(items)
