class AssetTracker:
    def __init__(self, initial_funds):
        self.positions = []
        self.cash = initial_funds

    def update_position(self, stock):
        for position in self.positions:
            if position['name'] == stock['name']:
                position['quantity'] += stock['quantity']
                return

        self.positions.append(stock)

    def decrease_position(self, stock):
        for position in self.positions:
            if position['name'] == stock['name'] and position['quantity'] >= stock['quantity']:
                position['quantity'] -= stock['quantity']
                if position['quantity'] == 0:
                    self.positions.remove(position)
                return True
        return False

    def purchase_asset(self, stock):
        if stock['price'] * stock['quantity'] > self.cash:
            return False
        else:
            self.update_position(stock)
            self.cash -= stock['price'] * stock['quantity']
            return True

    def sell_asset(self, stock):
        if not self.decrease_position(stock):
            return False
        self.cash += stock['price'] * stock['quantity']
        return True

    def total_worth(self):
        total_worth = self.cash
        for stock in self.positions:
            total_worth += stock['price'] * stock['quantity']
        return total_worth

    def prepare_report(self):
        report = []
        for stock in self.positions:
            stock_worth = self.calculate_asset_worth(stock)
            report.append({"name": stock["name"], "worth": stock_worth})
        portfolio_worth = self.total_worth()
        return portfolio_worth, report

    def calculate_asset_worth(self, stock):
        return stock['price'] * stock['quantity']
