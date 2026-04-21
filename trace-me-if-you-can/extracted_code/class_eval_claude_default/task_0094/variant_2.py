from collections import defaultdict

class VendingMachine:
    def __init__(self):
        self.inventory = defaultdict(lambda: {'price': 0, 'quantity': 0})
        self.balance = 0
        self._items_exist = set()

    def add_item(self, item_name, price, quantity):
        if item_name not in self._items_exist:
            self.inventory[item_name]['price'] = price
            self.inventory[item_name]['quantity'] = quantity
            self._items_exist.add(item_name)
        else:
            self.inventory[item_name]['quantity'] += quantity

    def insert_coin(self, amount):
        self.balance += amount
        return self.balance

    def purchase_item(self, item_name):
        if item_name not in self._items_exist:
            return False
        
        item = self.inventory[item_name]
        if item['quantity'] <= 0 or self.balance < item['price']:
            return False
            
        self.balance -= item['price']
        item['quantity'] -= 1
        return self.balance

    def restock_item(self, item_name, quantity):
        if item_name in self._items_exist:
            self.inventory[item_name]['quantity'] += quantity
            return True
        return False

    def display_items(self):
        if not self._items_exist:
            return False
        
        result = []
        for item_name in self._items_exist:
            item_info = self.inventory[item_name]
            result.append(f"{item_name} - ${item_info['price']} [{item_info['quantity']}]")
        return "\n".join(result)
