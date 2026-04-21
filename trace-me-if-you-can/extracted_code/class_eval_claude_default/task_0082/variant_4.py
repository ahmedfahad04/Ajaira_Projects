class StockPortfolioTracker:
    def __init__(self, cash_balance):
        self.portfolio = []
        self.cash_balance = cash_balance

    def add_stock(self, stock):
        try:
            existing_stock = next(pf for pf in self.portfolio if pf['name'] == stock['name'])
            existing_stock['quantity'] += stock['quantity']
        except StopIteration:
            self.portfolio.append(stock)

    def remove_stock(self, stock):
        try:
            existing_stock = next(pf for pf in self.portfolio 
                                if pf['name'] == stock['name'] and pf['quantity'] >= stock['quantity'])
            existing_stock['quantity'] -= stock['quantity']
            if existing_stock['quantity'] == 0:
                self.portfolio.remove(existing_stock)
            return True
        except StopIteration:
            return False

    def buy_stock(self, stock):
        total_cost = stock['price'] * stock['quantity']
        if total_cost <= self.cash_balance:
            self.add_stock(stock)
            self.cash_balance -= total_cost
            return True
        return False

    def sell_stock(self, stock):
        success = self.remove_stock(stock)
        if success:
            self.cash_balance += stock['price'] * stock['quantity']
        return success

    def calculate_portfolio_value(self):
        portfolio_stock_value = 0
        for stock in self.portfolio:
            portfolio_stock_value += self.get_stock_value(stock)
        return self.cash_balance + portfolio_stock_value

    def get_portfolio_summary(self):
        summary = []
        for stock in self.portfolio:
            value = self.get_stock_value(stock)
            summary.append({"name": stock["name"], "value": value})
        portfolio_value = self.calculate_portfolio_value()
        return portfolio_value, summary

    def get_stock_value(self, stock):
        return stock['price'] * stock['quantity']
