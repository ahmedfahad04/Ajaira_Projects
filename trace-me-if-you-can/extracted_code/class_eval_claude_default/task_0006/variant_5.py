class AvgPartition:
    def __init__(self, lst, limit):
        self.lst = lst
        self.limit = limit
        self.length = len(lst)

    def get(self, index):
        def partition_boundaries():
            base_size = self.length // self.limit
            extra = self.length % self.limit
            
            boundaries = [0]
            pos = 0
            
            for i in range(self.limit):
                pos += base_size + (1 if i < extra else 0)
                boundaries.append(pos)
                
            return boundaries
        
        bounds = partition_boundaries()
        return self.lst[bounds[index]:bounds[index + 1]]
