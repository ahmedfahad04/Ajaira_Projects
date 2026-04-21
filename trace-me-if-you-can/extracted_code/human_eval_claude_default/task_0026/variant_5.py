def filter_unique_occurrences(numbers):
    class FrequencyTracker:
        def __init__(self):
            self.frequencies = {}
        
        def add(self, item):
            self.frequencies[item] = self.frequencies.get(item, 0) + 1
        
        def is_unique_or_single(self, item):
            return self.frequencies.get(item, 0) <= 1
    
    tracker = FrequencyTracker()
    
    # First pass: count frequencies
    for num in numbers:
        tracker.add(num)
    
    # Second pass: filter based on frequency
    return list(filter(tracker.is_unique_or_single, numbers))
