def generate_prefixes(string):
    return list(map(lambda i: string[:i+1], range(len(string))))
