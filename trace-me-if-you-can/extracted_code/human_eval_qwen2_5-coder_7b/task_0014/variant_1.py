def generate_substrings(input_string):
    substrings = []
    for index in range(len(input_string)):
        substrings.append(input_string[0:index + 1])
    return substrings
