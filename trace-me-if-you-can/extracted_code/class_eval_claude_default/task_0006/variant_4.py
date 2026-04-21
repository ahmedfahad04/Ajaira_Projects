class AvgPartition:
    def __init__(self, lst, limit):
        self.lst = lst
        self.limit = limit

    def get(self, index):
        total_length = len(self.lst)
        quotient, remainder = divmod(total_length, self.limit)
        
        # Calculate cumulative lengths up to index
        cumulative_length = 0
        for i in range(index):
            cumulative_length += quotient + (1 if i < remainder else 0)
        
        start = cumulative_length
        current_partition_size = quotient + (1 if index < remainder else 0)
        end = start + current_partition_size
        
        return self.lst[start:end]
