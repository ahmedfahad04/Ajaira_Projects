def calculate_substring_appearances(main_string, target_substring):
    occurrence_count = 0
    for position in range(len(main_string) - len(target_substring) + 1):
        if main_string[position:position+len(target_substring)] == target_substring:
            occurrence_count += 1
    return occurrence_count
