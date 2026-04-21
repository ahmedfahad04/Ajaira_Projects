class InventoryManager:
       def __init__(self):
           self.items = {}
           self.balance = 0

       def add_product(self, product_name, price, quantity):
           if not self.update_stock(product_name, quantity):
               self.items[product_name] = {'price': price, 'quantity': quantity}

       def insert_money(self, amount):
           self.balance += amount
           return self.balance

       def buy_product(self, product_name):
           if product_name in self.items:
               item = self.items[product_name]
               if item['quantity'] > 0 and self.balance >= item['price']:
                   self.balance -= item['price']
                   item['quantity'] -= 1
                   return self.balance
               else:
                   return False
           else:
               return False

       def update_stock(self, product_name, quantity):
           if product_name in self.items:
               self.items[product_name]['quantity'] += quantity
               return True
           else:
               return False

       def list_products(self):
           if not self.items:
               return False
           else:
               products = []
               for product_name, product_info in self.items.items():
                   products.append(f"{product_name} - ${product_info['price']} [{product_info['quantity']}]")
               return "\n".join(products)
