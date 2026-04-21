class VendingMachine:
    def __init__(self):
        self.inventory = {}
        self.balance = 0

    def add_item(self, item_name, price, quantity):
        try:
            self.restock_item(item_name, quantity)
        except KeyError:
            self.inventory[item_name] = {'price': price, 'quantity': quantity}

    def insert_coin(self, amount):
        self.balance += amount
        return self.balance

    def purchase_item(self, item_name):
        try:
            item = self.inventory[item_name]
            if item['quantity'] > 0 and self.balance >= item['price']:
                self.balance -= item['price']
                item['quantity'] -= 1
                return self.balance
            return False
        except KeyError:
            return False

    def restock_item(self, item_name, quantity):
        try:
            self.inventory[item_name]['quantity'] += quantity
            return True
        except KeyError:
            raise KeyError(f"Item {item_name} not found")

    def display_items(self):
        if len(self.inventory) == 0:
            return False
        
        items = []
        for item_name in self.inventory:
            item_info = self.inventory[item_name]
            items.append(f"{item_name} - ${item_info['price']} [{item_info['quantity']}]")
        return "\n".join(items)
