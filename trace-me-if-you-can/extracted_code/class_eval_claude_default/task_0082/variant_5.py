class StockPortfolioTracker:
    def __init__(self, cash_balance):
        self.portfolio = []
        self.cash_balance = cash_balance

    def add_stock(self, stock):
        stock_names = [pf['name'] for pf in self.portfolio]
        if stock['name'] in stock_names:
            index = stock_names.index(stock['name'])
            self.portfolio[index]['quantity'] += stock['quantity']
        else:
            self.portfolio.append(stock)

    def remove_stock(self, stock):
        matching_stocks = [(i, pf) for i, pf in enumerate(self.portfolio) 
                          if pf['name'] == stock['name'] and pf['quantity'] >= stock['quantity']]
        
        if matching_stocks:
            index, matched_stock = matching_stocks[0]
            matched_stock['quantity'] -= stock['quantity']
            if matched_stock['quantity'] == 0:
                self.portfolio.pop(index)
            return True
        return False

    def buy_stock(self, stock):
        purchase_amount = stock['price'] * stock['quantity']
        has_sufficient_funds = purchase_amount <= self.cash_balance
        
        if has_sufficient_funds:
            self.add_stock(stock)
            self.cash_balance -= purchase_amount
            
        return has_sufficient_funds

    def sell_stock(self, stock):
        sale_amount = stock['price'] * stock['quantity']
        transaction_completed = self.remove_stock(stock)
        
        if transaction_completed:
            self.cash_balance += sale_amount
            
        return transaction_completed

    def calculate_portfolio_value(self):
        total_value = self.cash_balance
        stock_values = [self.get_stock_value(stock) for stock in self.portfolio]
        
        for value in stock_values:
            total_value += value
            
        return total_value

    def get_portfolio_summary(self):
        summary = [{"name": stock["name"], "value": self.get_stock_value(stock)} 
                  for stock in self.portfolio]
        portfolio_value = self.calculate_portfolio_value()
        return portfolio_value, summary

    def get_stock_value(self, stock):
        return stock['price'] * stock['quantity']
