def count_substring_occurrences(string, substring):
    if not substring or len(substring) > len(string):
        return 0
    
    matches = 0
    substring_len = len(substring)
    
    for start_idx in range(len(string) - substring_len + 1):
        match = True
        for char_idx in range(substring_len):
            if string[start_idx + char_idx] != substring[char_idx]:
                match = False
                break
        if match:
            matches += 1
    
    return matches
