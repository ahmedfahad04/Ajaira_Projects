def count_substring_occurrences(text, pattern):
    match_count = 0
    for index in range(len(text) - len(pattern) + 1):
        if text[index:index+len(pattern)] == pattern:
            match_count += 1
    return match_count
