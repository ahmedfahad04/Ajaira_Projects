from functools import reduce
from operator import add

class StockPortfolioTracker:
    def __init__(self, cash_balance):
        self.portfolio = []
        self.cash_balance = cash_balance

    def add_stock(self, stock):
        existing_stock = self._get_existing_stock(stock['name'])
        if existing_stock:
            existing_stock['quantity'] += stock['quantity']
        else:
            self.portfolio.append(stock)

    def _get_existing_stock(self, name):
        matches = [stock for stock in self.portfolio if stock['name'] == name]
        return matches[0] if matches else None

    def remove_stock(self, stock):
        existing_stock = self._get_existing_stock(stock['name'])
        if existing_stock and existing_stock['quantity'] >= stock['quantity']:
            existing_stock['quantity'] -= stock['quantity']
            if existing_stock['quantity'] == 0:
                self.portfolio.remove(existing_stock)
            return True
        return False

    def buy_stock(self, stock):
        required_funds = self.get_stock_value(stock)
        if required_funds <= self.cash_balance:
            self.add_stock(stock)
            self.cash_balance -= required_funds
            return True
        return False

    def sell_stock(self, stock):
        sale_successful = self.remove_stock(stock)
        if sale_successful:
            self.cash_balance += self.get_stock_value(stock)
        return sale_successful

    def calculate_portfolio_value(self):
        stock_values = map(self.get_stock_value, self.portfolio)
        return reduce(add, stock_values, self.cash_balance)

    def get_portfolio_summary(self):
        summary = [{"name": stock["name"], "value": self.get_stock_value(stock)} 
                  for stock in self.portfolio]
        portfolio_value = self.calculate_portfolio_value()
        return portfolio_value, summary

    def get_stock_value(self, stock):
        return stock['price'] * stock['quantity']
