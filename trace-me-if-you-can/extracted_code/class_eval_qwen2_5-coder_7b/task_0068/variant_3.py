class PaginationUtil:
    def __init__(self, collection, items_per_group):
        self.collection = collection
        self.items_per_group = items_per_group
        self.total_items = len(collection)
        self.total_groups = -(-self.total_items // items_per_group)  # Ceiling division

    def extract_group(self, group_number):
        if group_number < 1 or group_number > self.total_groups:
            return []

        start = (group_number - 1) * self.items_per_group
        end = start + self.items_per_group
        return self.collection[start:end]

    def get_group_info(self, group_number):
        if group_number < 1 or group_number > self.total_groups:
            return {}

        start = (group_number - 1) * self.items_per_group
        end = min(start + self.items_per_group, self.total_items)
        group_data = self.collection[start:end]

        group_details = {
            "group_number": group_number,
            "items_per_group": self.items_per_group,
            "total_groups": self.total_groups,
            "total_items": self.total_items,
            "has_previous": group_number > 1,
            "has_next": group_number < self.total_groups,
            "data": group_data
        }
        return group_details

    def find_items(self, search_term):
        matched_items = [item for item in self.collection if search_term in str(item)]
        matched_count = len(matched_items)
        matched_pages = -(-matched_count // self.items_per_group)  # Ceiling division

        find_details = {
            "search_term": search_term,
            "total_matches": matched_count,
            "total_pages": matched_pages,
            "matched_items": matched_items
        }
        return find_details
