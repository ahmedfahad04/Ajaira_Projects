class StockPortfolioTracker:
    def __init__(self, cash_balance):
        self.portfolio = []
        self.cash_balance = cash_balance

    def _find_stock_index(self, stock_name):
        return next((i for i, stock in enumerate(self.portfolio) if stock['name'] == stock_name), -1)

    def add_stock(self, stock):
        index = self._find_stock_index(stock['name'])
        if index != -1:
            self.portfolio[index]['quantity'] += stock['quantity']
        else:
            self.portfolio.append(stock)

    def remove_stock(self, stock):
        index = self._find_stock_index(stock['name'])
        if index != -1 and self.portfolio[index]['quantity'] >= stock['quantity']:
            self.portfolio[index]['quantity'] -= stock['quantity']
            if self.portfolio[index]['quantity'] == 0:
                self.portfolio.pop(index)
            return True
        return False

    def buy_stock(self, stock):
        transaction_cost = stock['price'] * stock['quantity']
        if transaction_cost <= self.cash_balance:
            self.add_stock(stock)
            self.cash_balance -= transaction_cost
            return True
        return False

    def sell_stock(self, stock):
        if self.remove_stock(stock):
            self.cash_balance += stock['price'] * stock['quantity']
            return True
        return False

    def calculate_portfolio_value(self):
        stock_values = [self.get_stock_value(stock) for stock in self.portfolio]
        return self.cash_balance + sum(stock_values)

    def get_portfolio_summary(self):
        summary = []
        for stock in self.portfolio:
            summary.append({"name": stock["name"], "value": self.get_stock_value(stock)})
        return self.calculate_portfolio_value(), summary

    def get_stock_value(self, stock):
        return stock['price'] * stock['quantity']
