class VendingMachine:
    def __init__(self):
        self.inventory = {}
        self.balance = 0

    def add_item(self, item_name, price, quantity):
        self.inventory[item_name] = self.inventory.get(item_name, {'price': price, 'quantity': 0})
        if self.inventory[item_name]['quantity'] == 0:
            self.inventory[item_name]['price'] = price
        self.inventory[item_name]['quantity'] += quantity

    def insert_coin(self, amount):
        self.balance += amount
        return self.balance

    def purchase_item(self, item_name):
        item = self.inventory.get(item_name)
        if item and item['quantity'] > 0 and self.balance >= item['price']:
            self.balance -= item['price']
            item['quantity'] -= 1
            return self.balance
        return False

    def restock_item(self, item_name, quantity):
        if item_name in self.inventory:
            self.inventory[item_name]['quantity'] += quantity
            return True
        return False

    def display_items(self):
        return "\n".join([f"{name} - ${info['price']} [{info['quantity']}]" 
                         for name, info in self.inventory.items()]) if self.inventory else False
