class PageHelper:
    def __init__(self, items, items_per_page):
        self.items = items
        self.items_per_page = items_per_page
        self.item_count = len(items)
        self.page_count = (self.item_count + items_per_page - 1) // items_per_page

    def retrieve_page(self, page_number):
        if page_number < 1 or page_number > self.page_count:
            return []

        start = (page_number - 1) * self.items_per_page
        end = start + self.items_per_page
        return self.items[start:end]

    def get_page_details(self, page_number):
        if page_number < 1 or page_number > self.page_count:
            return {}

        start = (page_number - 1) * self.items_per_page
        end = min(start + self.items_per_page, self.item_count)
        page_data = self.items[start:end]

        details = {
            "current_page": page_number,
            "per_page": self.items_per_page,
            "total_pages": self.page_count,
            "total_items": self.item_count,
            "has_previous": page_number > 1,
            "has_next": page_number < self.page_count,
            "data": page_data
        }
        return details

    def search_items(self, term):
        matches = [item for item in self.items if term in str(item)]
        match_count = len(matches)
        match_pages = (match_count + self.items_per_page - 1) // self.items_per_page

        search_details = {
            "term": term,
            "total_matches": match_count,
            "total_pages": match_pages,
            "matches": matches
        }
        return search_details
