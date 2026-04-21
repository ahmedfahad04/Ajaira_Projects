def matches_prefix(string):
    return string.startswith(prefix)

return list(filter(matches_prefix, strings))
