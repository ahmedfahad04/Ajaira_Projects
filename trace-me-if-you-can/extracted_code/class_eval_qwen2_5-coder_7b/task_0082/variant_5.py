class AssetHolder:
    def __init__(self, initial_cash):
        self.positions = []
        self.cash_balance = initial_cash

    def add_stock_position(self, stock):
        for position in self.positions:
            if position['name'] == stock['name']:
                position['quantity'] += stock['quantity']
                return

        self.positions.append(stock)

    def reduce_stock_position(self, stock):
        for position in self.positions:
            if position['name'] == stock['name'] and position['quantity'] >= stock['quantity']:
                position['quantity'] -= stock['quantity']
                if position['quantity'] == 0:
                    self.positions.remove(position)
                return True
        return False

    def purchase_stock(self, stock):
        if stock['price'] * stock['quantity'] > self.cash_balance:
            return False
        else:
            self.add_stock_position(stock)
            self.cash_balance -= stock['price'] * stock['quantity']
            return True

    def sell_stock(self, stock):
        if not self.reduce_stock_position(stock):
            return False
        self.cash_balance += stock['price'] * stock['quantity']
        return True

    def calculate_total_portfolio_value(self):
        total_value = self.cash_balance
        for stock in self.positions:
            total_value += stock['price'] * stock['quantity']
        return total_value

    def generate_portfolio_summary(self):
        summary = []
        for stock in self.positions:
            stock_value = self.calculate_stock_value(stock)
            summary.append({"name": stock["name"], "value": stock_value})
        portfolio_value = self.calculate_total_portfolio_value()
        return portfolio_value, summary

    def calculate_stock_value(self, stock):
        return stock['price'] * stock['quantity']
