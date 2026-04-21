class DataPartitioner:
    def __init__(self, data_list, partition_limit):
        self.data_list = data_list
        self.partition_limit = partition_limit

    def compute_partition_details(self):
        partition_base_size = len(self.data_list) // self.partition_limit
        partition_remainder = len(self.data_list) % self.partition_limit
        return partition_base_size, partition_remainder

    def fetch_partition(self, partition_index):
        base_size, remainder = self.compute_partition_details()
        start = partition_index * base_size + min(partition_index, remainder)
        end = start + base_size
        if partition_index + 1 <= remainder:
            end += 1
        return self.data_list[start:end]
