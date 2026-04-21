class PageUtil:
    def __init__(self, data, page_size):
        if page_size <= 0:
            raise ValueError("Page size must be positive")
        
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = -(-self.total_items // page_size)  # Ceiling division trick

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _is_page_valid(self, page_number):
        try:
            return isinstance(page_number, int) and 1 <= page_number <= self.total_pages
        except (TypeError, ValueError):
            return False

    def get_page(self, page_number):
        with self:
            if not self._is_page_valid(page_number):
                return []
            
            offset = (page_number - 1) * self.page_size
            limit = offset + self.page_size
            return self.data[offset:limit]

    def get_page_info(self, page_number):
        with self:
            if not self._is_page_valid(page_number):
                return {}

            offset = (page_number - 1) * self.page_size
            limit = min(offset + self.page_size, self.total_items)
            current_page_data = self.data[offset:limit]

            return dict([
                ("current_page", page_number),
                ("per_page", self.page_size),
                ("total_pages", self.total_pages),
                ("total_items", self.total_items),
                ("has_previous", page_number > 1),
                ("has_next", page_number < self.total_pages),
                ("data", current_page_data)
            ])

    def search(self, keyword):
        with self:
            search_results = []
            for element in self.data:
                string_representation = str(element)
                if keyword in string_representation:
                    search_results.append(element)
            
            result_total = len(search_results)
            page_count = -(-result_total // self.page_size)  # Ceiling division
            
            search_summary = {}
            search_summary["keyword"] = keyword
            search_summary["total_results"] = result_total
            search_summary["total_pages"] = page_count
            search_summary["results"] = search_results
            return search_summary
