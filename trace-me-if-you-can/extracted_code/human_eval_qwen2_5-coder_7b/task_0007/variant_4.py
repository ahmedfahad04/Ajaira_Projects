def find_substrings(string_list, sub):
    matches = []
    for string in string_list:
        if sub in string:
            matches.append(string)
    return matches
