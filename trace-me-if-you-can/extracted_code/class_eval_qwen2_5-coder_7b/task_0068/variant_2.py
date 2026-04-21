class PageManager:
    def __init__(self, dataset, page_limit):
        self.dataset = dataset
        self.page_limit = page_limit
        self.item_count = len(dataset)
        self.page_count = (self.item_count + page_limit - 1) // page_limit

    def fetch_page(self, page):
        if page < 1 or page > self.page_count:
            return {"data": [], "details": {}}

        start = (page - 1) * self.page_limit
        end = min(start + self.page_limit, self.item_count)
        page_data = self.dataset[start:end]

        details = {
            "current_page": page,
            "page_limit": self.page_limit,
            "total_pages": self.page_count,
            "total_items": self.item_count,
            "has_previous": page > 1,
            "has_next": page < self.page_count,
            "data": page_data
        }
        return {"data": page_data, "details": details}

    def locate_items(self, keyword):
        found_items = [item for item in self.dataset if keyword in str(item)]
        found_count = len(found_items)
        found_pages = (found_count + self.page_limit - 1) // self.page_limit

        locate_details = {
            "keyword": keyword,
            "total_found": found_count,
            "total_pages": found_pages,
            "found_items": found_items
        }
        return locate_details
