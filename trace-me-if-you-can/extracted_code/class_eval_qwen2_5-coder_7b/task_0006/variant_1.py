class AveragePartition:
    def __init__(self, collection, partition_size):
        self.collection = collection
        self.partition_size = partition_size

    def determine_partition_size(self):
        base_size = len(self.collection) // self.partition_size
        extra_elements = len(self.collection) % self.partition_size
        return base_size, extra_elements

    def retrieve_partition(self, partition_index):
        base_size, extra_elements = self.determine_partition_size()
        start_index = partition_index * base_size + min(partition_index, extra_elements)
        end_index = start_index + base_size
        if partition_index + 1 <= extra_elements:
            end_index += 1
        return self.collection[start_index:end_index]
