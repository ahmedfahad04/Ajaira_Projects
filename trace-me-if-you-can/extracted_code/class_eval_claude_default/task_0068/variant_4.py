class PageUtil:
    def __init__(self, data, page_size):
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = (self.total_items + page_size - 1) // page_size

    def _page_iterator(self, page_number):
        """Generator that yields items for a specific page"""
        if page_number < 1 or page_number > self.total_pages:
            return
        
        start_idx = (page_number - 1) * self.page_size
        end_idx = min(start_idx + self.page_size, self.total_items)
        
        for i in range(start_idx, end_idx):
            yield self.data[i]

    def get_page(self, page_number):
        return list(self._page_iterator(page_number))

    def get_page_info(self, page_number):
        page_items = list(self._page_iterator(page_number))
        
        if not page_items and (page_number < 1 or page_number > self.total_pages):
            return {}

        navigation_info = {
            "current_page": page_number,
            "per_page": self.page_size,
            "total_pages": self.total_pages,
            "total_items": self.total_items
        }
        
        pagination_state = {
            "has_previous": page_number > 1,
            "has_next": page_number < self.total_pages,
            "data": page_items
        }
        
        return {**navigation_info, **pagination_state}

    def _search_generator(self, keyword):
        """Generator that yields matching items"""
        for item in self.data:
            item_str = str(item)
            if keyword in item_str:
                yield item

    def search(self, keyword):
        found_items = list(self._search_generator(keyword))
        items_found = len(found_items)
        
        pages_required = 0 if items_found == 0 else (items_found + self.page_size - 1) // self.page_size
        
        search_metadata = {
            "keyword": keyword,
            "total_results": items_found,
            "total_pages": pages_required,
            "results": found_items
        }
        return search_metadata
