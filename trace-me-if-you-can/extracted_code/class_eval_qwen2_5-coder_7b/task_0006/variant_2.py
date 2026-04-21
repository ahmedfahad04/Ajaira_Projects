class Divider:
    def __init__(self, array, division_count):
        self.array = array
        self.division_count = division_count

    def calculate_partition_size(self):
        fixed_size = len(self.array) // self.division_count
        extra_parts = len(self.array) % self.division_count
        return fixed_size, extra_parts

    def extract_partition(self, partition_number):
        fixed_size, extra_parts = self.calculate_partition_size()
        start = partition_number * fixed_size + min(partition_number, extra_parts)
        end = start + fixed_size
        if partition_number + 1 <= extra_parts:
            end += 1
        return self.array[start:end]
