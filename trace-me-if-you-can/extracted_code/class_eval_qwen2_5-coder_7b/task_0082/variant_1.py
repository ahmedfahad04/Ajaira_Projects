class StockHolder:
    def __init__(self, initial_cash):
        self.positions = []
        self.cash = initial_cash

    def include_stock(self, new_stock):
        for position in self.positions:
            if position['name'] == new_stock['name']:
                position['quantity'] += new_stock['quantity']
                return

        self.positions.append(new_stock)

    def deduct_stock(self, stock):
        for position in self.positions:
            if position['name'] == stock['name'] and position['quantity'] >= stock['quantity']:
                position['quantity'] -= stock['quantity']
                if position['quantity'] == 0:
                    self.positions.remove(position)
                return True
        return False

    def purchase_stock(self, stock):
        if stock['price'] * stock['quantity'] > self.cash:
            return False
        else:
            self.include_stock(stock)
            self.cash -= stock['price'] * stock['quantity']
            return True

    def sell_off_stock(self, stock):
        if not self.deduct_stock(stock):
            return False
        self.cash += stock['price'] * stock['quantity']
        return True

    def total_portfolio_worth(self):
        total_worth = self.cash
        for stock in self.positions:
            total_worth += stock['price'] * stock['quantity']
        return total_worth

    def prepare_portfolio_report(self):
        report = []
        for stock in self.positions:
            stock_worth = self.calculate_stock_worth(stock)
            report.append({"name": stock["name"], "worth": stock_worth})
        portfolio_worth = self.total_portfolio_worth()
        return portfolio_worth, report

    def calculate_stock_worth(self, stock):
        return stock['price'] * stock['quantity']
