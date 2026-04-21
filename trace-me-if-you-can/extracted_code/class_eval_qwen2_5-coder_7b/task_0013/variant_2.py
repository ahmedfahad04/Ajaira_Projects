class LibrarySystem:
        def __init__(self):
            self.book_list = {}

        def acquire_book(self, title, quantity=1):
            if title in self.book_list:
                self.book_list[title] += quantity
            else:
                self.book_list[title] = quantity

        def release_book(self, title, quantity):
            if title not in self.book_list or self.book_list[title] < quantity:
                raise Exception("Book not available or insufficient quantity")
            self.book_list[title] -= quantity
            if self.book_list[title] == 0:
                del self.book_list[title]

        def show_inventory(self):
            return self.book_list

        def get_book_count(self, title):
            return self.book_list.get(title, 0)
