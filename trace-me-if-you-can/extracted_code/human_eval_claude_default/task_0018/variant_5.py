def count_substring_occurrences(string, substring):
    occurrences = 0
    window_size = len(substring)
    
    for end in range(window_size, len(string) + 1):
        start = end - window_size
        if string[start:end] == substring:
            occurrences += 1
    
    return occurrences
