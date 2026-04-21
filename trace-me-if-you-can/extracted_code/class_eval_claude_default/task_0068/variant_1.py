import math
from typing import List, Dict, Any

class PageUtil:
    def __init__(self, data: List[Any], page_size: int):
        self.data = data
        self.page_size = page_size
        self.total_items = len(data)
        self.total_pages = math.ceil(self.total_items / page_size)

    @staticmethod
    def _is_valid_page(page_number: int, total_pages: int) -> bool:
        return 1 <= page_number <= total_pages

    @staticmethod
    def _calculate_slice_bounds(page_number: int, page_size: int) -> tuple:
        start = (page_number - 1) * page_size
        end = start + page_size
        return start, end

    def get_page(self, page_number: int) -> List[Any]:
        return (self.data[slice(*self._calculate_slice_bounds(page_number, self.page_size))] 
                if self._is_valid_page(page_number, self.total_pages) else [])

    def get_page_info(self, page_number: int) -> Dict[str, Any]:
        if not self._is_valid_page(page_number, self.total_pages):
            return {}
        
        start, end = self._calculate_slice_bounds(page_number, self.page_size)
        page_data = self.data[start:min(end, self.total_items)]
        
        return {
            "current_page": page_number,
            "per_page": self.page_size,
            "total_pages": self.total_pages,
            "total_items": self.total_items,
            "has_previous": page_number > 1,
            "has_next": page_number < self.total_pages,
            "data": page_data
        }

    def search(self, keyword: str) -> Dict[str, Any]:
        filtered_results = list(filter(lambda item: keyword in str(item), self.data))
        result_count = len(filtered_results)
        
        return {
            "keyword": keyword,
            "total_results": result_count,
            "total_pages": math.ceil(result_count / self.page_size) if result_count > 0 else 0,
            "results": filtered_results
        }
