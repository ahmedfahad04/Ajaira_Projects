class Pagination:
    def __init__(self, data, items_per_page):
        self.data = data
        self.items_per_page = items_per_page
        self.item_count = len(data)
        self.page_count = (self.item_count + items_per_page - 1) // items_per_page

    def get_page_data(self, page_number):
        if page_number < 1 or page_number > self.page_count:
            return {"data": [], "details": {}}

        start = (page_number - 1) * self.items_per_page
        end = start + self.items_per_page
        page_data = self.data[start:end]

        details = {
            "current_page": page_number,
            "per_page": self.items_per_page,
            "total_pages": self.page_count,
            "total_items": self.item_count,
            "has_previous": page_number > 1,
            "has_next": page_number < self.page_count,
            "data": page_data
        }
        return {"data": page_data, "details": details}

    def locate_items(self, keyword):
        matches = [item for item in self.data if keyword in str(item)]
        match_count = len(matches)
        match_pages = (match_count + self.items_per_page - 1) // self.items_per_page

        locate_details = {
            "keyword": keyword,
            "total_matches": match_count,
            "total_pages": match_pages,
            "matches": matches
        }
        return locate_details
