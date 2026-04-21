def count_substring_occurrences(string, substring):
    def count_from_position(text, pattern, pos):
        if pos > len(text) - len(pattern):
            return 0
        
        current_match = 1 if text[pos:pos+len(pattern)] == pattern else 0
        return current_match + count_from_position(text, pattern, pos + 1)
    
    return count_from_position(string, substring, 0)
