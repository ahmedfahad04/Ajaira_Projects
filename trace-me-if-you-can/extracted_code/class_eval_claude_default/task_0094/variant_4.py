class VendingMachine:
    def __init__(self):
        self.inventory = {}
        self.balance = 0

    def _item_exists(self, item_name):
        return item_name in self.inventory

    def _can_purchase(self, item_name):
        return (self._item_exists(item_name) and 
                self.inventory[item_name]['quantity'] > 0 and 
                self.balance >= self.inventory[item_name]['price'])

    def add_item(self, item_name, price, quantity):
        if self._item_exists(item_name):
            self.inventory[item_name]['quantity'] += quantity
        else:
            self.inventory[item_name] = {'price': price, 'quantity': quantity}

    def insert_coin(self, amount):
        self.balance += amount
        return self.balance

    def purchase_item(self, item_name):
        if not self._can_purchase(item_name):
            return False
        
        item = self.inventory[item_name]
        self.balance -= item['price']
        item['quantity'] -= 1
        return self.balance

    def restock_item(self, item_name, quantity):
        if not self._item_exists(item_name):
            return False
        
        self.inventory[item_name]['quantity'] += quantity
        return True

    def display_items(self):
        if not bool(self.inventory):
            return False
        
        return "\n".join(f"{name} - ${data['price']} [{data['quantity']}]" 
                        for name, data in self.inventory.items())
