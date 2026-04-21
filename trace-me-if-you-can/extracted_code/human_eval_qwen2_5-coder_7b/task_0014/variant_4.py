def substring_collection(input_str):
    substrings = []
    for pos in range(len(input_str)):
        substrings.append(input_str[0:pos + 1])
    return substrings
