class StockPortfolioTracker:
    def __init__(self, cash_balance):
        self.portfolio = {}
        self.cash_balance = cash_balance

    def add_stock(self, stock):
        name = stock['name']
        if name in self.portfolio:
            self.portfolio[name]['quantity'] += stock['quantity']
        else:
            self.portfolio[name] = stock.copy()

    def remove_stock(self, stock):
        name = stock['name']
        if name in self.portfolio and self.portfolio[name]['quantity'] >= stock['quantity']:
            self.portfolio[name]['quantity'] -= stock['quantity']
            if self.portfolio[name]['quantity'] == 0:
                del self.portfolio[name]
            return True
        return False

    def buy_stock(self, stock):
        cost = stock['price'] * stock['quantity']
        if cost > self.cash_balance:
            return False
        self.add_stock(stock)
        self.cash_balance -= cost
        return True

    def sell_stock(self, stock):
        if not self.remove_stock(stock):
            return False
        self.cash_balance += stock['price'] * stock['quantity']
        return True

    def calculate_portfolio_value(self):
        return self.cash_balance + sum(self.get_stock_value(stock) for stock in self.portfolio.values())

    def get_portfolio_summary(self):
        summary = [{"name": name, "value": self.get_stock_value(stock)} 
                  for name, stock in self.portfolio.items()]
        portfolio_value = self.calculate_portfolio_value()
        return portfolio_value, summary

    def get_stock_value(self, stock):
        return stock['price'] * stock['quantity']
