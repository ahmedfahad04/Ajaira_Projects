class PageUtil:
    def __init__(self, data, page_size):
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = self._compute_total_pages()

    def _compute_total_pages(self):
        return (self.total_items + self.page_size - 1) // self.page_size

    def _execute_page_validation(self, page_number):
        return page_number >= 1 and page_number <= self.total_pages

    def _execute_index_calculation(self, page_number):
        base_index = (page_number - 1) * self.page_size
        return base_index, base_index + self.page_size

    def get_page(self, page_number):
        return (self._extract_page_slice(page_number) 
                if self._execute_page_validation(page_number) 
                else [])

    def _extract_page_slice(self, page_number):
        start_pos, end_pos = self._execute_index_calculation(page_number)
        return self.data[start_pos:end_pos]

    def get_page_info(self, page_number):
        if not self._execute_page_validation(page_number):
            return {}
        
        return self._build_page_metadata(page_number)

    def _build_page_metadata(self, page_number):
        start_pos, end_pos = self._execute_index_calculation(page_number)
        bounded_end = min(end_pos, self.total_items)
        page_content = self.data[start_pos:bounded_end]
        
        metadata = {}
        metadata = self._add_basic_info(metadata, page_number)
        metadata = self._add_navigation_info(metadata, page_number)
        metadata = self._add_content_info(metadata, page_content)
        return metadata

    def _add_basic_info(self, metadata, page_number):
        metadata["current_page"] = page_number
        metadata["per_page"] = self.page_size
        metadata["total_pages"] = self.total_pages
        metadata["total_items"] = self.total_items
        return metadata

    def _add_navigation_info(self, metadata, page_number):
        metadata["has_previous"] = page_number > 1
        metadata["has_next"] = page_number < self.total_pages
        return metadata

    def _add_content_info(self, metadata, page_content):
        metadata["data"] = page_content
        return metadata

    def search(self, keyword):
        matches = self._execute_search_filter(keyword)
        return self._build_search_results(keyword, matches)

    def _execute_search_filter(self, keyword):
        return [item for item in self.data if self._item_matches_keyword(item, keyword)]

    def _item_matches_keyword(self, item, keyword):
        return keyword in str(item)

    def _build_search_results(self, keyword, matches):
        match_count = len(matches)
        page_count = (match_count + self.page_size - 1) // self.page_size
        
        results = {}
        results["keyword"] = keyword
        results["total_results"] = match_count
        results["total_pages"] = page_count
        results["results"] = matches
        return results
