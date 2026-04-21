def running_maximum(numbers):
    if not numbers:
        return []
    
    def build_running_max(idx, prev_max):
        if idx >= len(numbers):
            return []
        
        current_max = max(prev_max, numbers[idx]) if prev_max is not None else numbers[idx]
        return [current_max] + build_running_max(idx + 1, current_max)
    
    return build_running_max(0, None)
