def generate_prefixes(string):
    return list(string[:i+1] for i in range(len(string)))
