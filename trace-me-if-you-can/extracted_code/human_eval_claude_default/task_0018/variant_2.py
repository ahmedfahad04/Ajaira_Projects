def count_substring_occurrences(string, substring):
    return sum(1 for i in range(len(string) - len(substring) + 1) 
               if string[i:i+len(substring)] == substring)
