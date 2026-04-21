class PageHandler:
    def __init__(self, items, items_per_page):
        self.items = items
        self.items_per_page = items_per_page
        self.total_items = len(items)
        self.total_pages = (self.total_items + self.items_per_page - 1) // self.items_per_page

    def get_data_for_page(self, page_number):
        if page_number < 1 or page_number > self.total_pages:
            return {"data": [], "details": {}}

        start_index = (page_number - 1) * self.items_per_page
        end_index = min(start_index + self.items_per_page, self.total_items)
        page_data = self.items[start_index:end_index]

        details = {
            "page_number": page_number,
            "items_per_page": self.items_per_page,
            "total_pages": self.total_pages,
            "total_items": self.total_items,
            "has_previous": page_number > 1,
            "has_next": page_number < self.total_pages,
            "data": page_data
        }
        return {"data": page_data, "details": details}

    def search_items(self, search_term):
        matches = [item for item in self.items if search_term in str(item)]
        match_count = len(matches)
        match_pages = (match_count + self.items_per_page - 1) // self.items_per_page

        search_details = {
            "search_term": search_term,
            "total_matches": match_count,
            "total_pages": match_pages,
            "matches": matches
        }
        return search_details
