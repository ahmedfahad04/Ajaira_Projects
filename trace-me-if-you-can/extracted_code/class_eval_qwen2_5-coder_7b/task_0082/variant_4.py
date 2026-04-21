class InvestmentTracker:
    def __init__(self, initial_cash):
        self.positions = []
        self.balance = initial_cash

    def update_stock(self, stock):
        for position in self.positions:
            if position['name'] == stock['name']:
                position['quantity'] += stock['quantity']
                return

        self.positions.append(stock)

    def reduce_stock(self, stock):
        for position in self.positions:
            if position['name'] == stock['name'] and position['quantity'] >= stock['quantity']:
                position['quantity'] -= stock['quantity']
                if position['quantity'] == 0:
                    self.positions.remove(position)
                return True
        return False

    def buy_stock(self, stock):
        if stock['price'] * stock['quantity'] > self.balance:
            return False
        else:
            self.update_stock(stock)
            self.balance -= stock['price'] * stock['quantity']
            return True

    def sell_stock(self, stock):
        if not self.reduce_stock(stock):
            return False
        self.balance += stock['price'] * stock['quantity']
        return True

    def calculate_total_value(self):
        total_value = self.balance
        for stock in self.positions:
            total_value += stock['price'] * stock['quantity']
        return total_value

    def create_portfolio_report(self):
        report = []
        for stock in self.positions:
            stock_value = self.compute_stock_value(stock)
            report.append({"name": stock["name"], "value": stock_value})
        portfolio_value = self.calculate_total_value()
        return portfolio_value, report

    def compute_stock_value(self, stock):
        return stock['price'] * stock['quantity']
