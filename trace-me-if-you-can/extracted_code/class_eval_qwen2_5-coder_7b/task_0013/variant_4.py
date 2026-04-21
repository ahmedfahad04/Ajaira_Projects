class BookTracker:
        def __init__(self):
            self.books = {}

        def add_copy(self, title, quantity=1):
            if title in self.books:
                self.books[title] += quantity
            else:
                self.books[title] = quantity

        def borrow_copy(self, title, quantity):
            if title not in self.books or self.books[title] < quantity:
                raise KeyError("Book unavailable or insufficient quantity")
            self.books[title] -= quantity
            if self.books[title] == 0:
                del self.books[title]

        def view_stock(self):
            return self.books

        def get_book_stock(self, title):
            return self.books.get(title, 0)
