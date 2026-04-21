class ElementGrouper:
    def __init__(self, element_list, group_count):
        self.element_list = element_list
        self.group_count = group_count

    def determine_group_size(self):
        main_size = len(self.element_list) // self.group_count
        extra_groups = len(self.element_list) % self.group_count
        return main_size, extra_groups

    def retrieve_group(self, group_index):
        main_size, extra_groups = self.determine_group_size()
        start_index = group_index * main_size + min(group_index, extra_groups)
        end_index = start_index + main_size
        if group_index + 1 <= extra_groups:
            end_index += 1
        return self.element_list[start_index:end_index]
