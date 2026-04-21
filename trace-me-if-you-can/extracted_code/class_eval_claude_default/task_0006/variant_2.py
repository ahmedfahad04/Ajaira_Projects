class AvgPartition:
    def __init__(self, lst, limit):
        self.partitions = self._create_partitions(lst, limit)
    
    def _create_partitions(self, lst, limit):
        size = len(lst) // limit
        remainder = len(lst) % limit
        partitions = []
        start = 0
        
        for i in range(limit):
            partition_size = size + (1 if i < remainder else 0)
            partitions.append(lst[start:start + partition_size])
            start += partition_size
            
        return partitions
    
    def get(self, index):
        return self.partitions[index]
