def find_longest_string(input_list):
    if not input_list:
        return None

    longest = max(input_list, key=len)
    return longest

result = find_longest_string(strings)
return result
