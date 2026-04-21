class PortfolioManager:
    def __init__(self, starting_cash):
        self.stocks = []
        self.balance = starting_cash

    def add_position(self, stock):
        for entry in self.stocks:
            if entry['name'] == stock['name']:
                entry['quantity'] += stock['quantity']
                return

        self.stocks.append(stock)

    def reduce_position(self, stock):
        for entry in self.stocks:
            if entry['name'] == stock['name'] and entry['quantity'] >= stock['quantity']:
                entry['quantity'] -= stock['quantity']
                if entry['quantity'] == 0:
                    self.stocks.remove(entry)
                return True
        return False

    def buy_stock(self, stock):
        if stock['price'] * stock['quantity'] > self.balance:
            return False
        else:
            self.add_position(stock)
            self.balance -= stock['price'] * stock['quantity']
            return True

    def sell_stock(self, stock):
        if not self.reduce_position(stock):
            return False
        self.balance += stock['price'] * stock['quantity']
        return True

    def get_total_value(self):
        total_value = self.balance
        for stock in self.stocks:
            total_value += stock['price'] * stock['quantity']
        return total_value

    def generate_portfolio_report(self):
        report = []
        for stock in self.stocks:
            stock_value = self.compute_stock_value(stock)
            report.append({"name": stock["name"], "value": stock_value})
        portfolio_value = self.get_total_value()
        return portfolio_value, report

    def compute_stock_value(self, stock):
        return stock['price'] * stock['quantity']
