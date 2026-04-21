class PageUtil:
    def __init__(self, data, page_size):
        self._data = data
        self._page_size = page_size

    @property
    def data(self):
        return self._data

    @property
    def page_size(self):
        return self._page_size

    @property
    def total_items(self):
        return len(self._data)

    @property
    def total_pages(self):
        return (self.total_items + self.page_size - 1) // self.page_size

    def _validate_page_bounds(self, page_number):
        return page_number >= 1 and page_number <= self.total_pages

    def _get_page_boundaries(self, page_number):
        start = (page_number - 1) * self.page_size
        return start, start + self.page_size

    def get_page(self, page_number):
        if not self._validate_page_bounds(page_number):
            return []
        
        start, end = self._get_page_boundaries(page_number)
        return self.data[start:end]

    def get_page_info(self, page_number):
        if not self._validate_page_bounds(page_number):
            return {}

        start, end = self._get_page_boundaries(page_number)
        actual_end = min(end, self.total_items)
        
        info = {}
        info.update({
            "current_page": page_number,
            "per_page": self.page_size,
            "total_pages": self.total_pages,
            "total_items": self.total_items
        })
        info.update({
            "has_previous": page_number > 1,
            "has_next": page_number < self.total_pages,
            "data": self.data[start:actual_end]
        })
        return info

    def search(self, keyword):
        matching_items = []
        for item in self.data:
            if keyword in str(item):
                matching_items.append(item)
        
        match_count = len(matching_items)
        pages_needed = (match_count + self.page_size - 1) // self.page_size
        
        return {
            "keyword": keyword,
            "total_results": match_count,
            "total_pages": pages_needed,
            "results": matching_items
        }
