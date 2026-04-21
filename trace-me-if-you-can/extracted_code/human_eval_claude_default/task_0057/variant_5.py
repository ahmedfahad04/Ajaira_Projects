def is_monotonic(l):
    non_increasing = non_decreasing = 0
    
    for i in range(len(l) - 1):
        if l[i] > l[i + 1]:
            non_decreasing += 1
        elif l[i] < l[i + 1]:
            non_increasing += 1
    
    return non_increasing == 0 or non_decreasing == 0
