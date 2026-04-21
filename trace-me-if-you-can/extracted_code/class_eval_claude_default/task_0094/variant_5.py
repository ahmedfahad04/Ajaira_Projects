class VendingMachine:
    def __init__(self):
        self.inventory = {}
        self.balance = 0

    def add_item(self, item_name, price, quantity):
        existing_item = self.inventory.get(item_name)
        if existing_item is None:
            self.inventory[item_name] = {'price': price, 'quantity': quantity}
        else:
            existing_item['quantity'] += quantity

    def insert_coin(self, amount):
        return self._update_balance(amount)

    def _update_balance(self, amount):
        self.balance += amount
        return self.balance

    def purchase_item(self, item_name):
        item_data = self.inventory.get(item_name)
        
        purchase_conditions = [
            item_data is not None,
            item_data and item_data['quantity'] > 0,
            item_data and self.balance >= item_data['price']
        ]
        
        if not all(purchase_conditions):
            return False
            
        self.balance -= item_data['price']
        item_data['quantity'] -= 1
        return self.balance

    def restock_item(self, item_name, quantity):
        target_item = self.inventory.get(item_name)
        if target_item is None:
            return False
        target_item['quantity'] += quantity
        return True

    def display_items(self):
        inventory_items = list(self.inventory.items())
        if len(inventory_items) == 0:
            return False
        
        formatted_items = [self._format_item(name, info) for name, info in inventory_items]
        return "\n".join(formatted_items)
    
    def _format_item(self, name, info):
        return f"{name} - ${info['price']} [{info['quantity']}]"
