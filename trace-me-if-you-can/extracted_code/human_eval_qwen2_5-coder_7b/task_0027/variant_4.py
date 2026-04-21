def change_case(s):
    return ''.join([c.upper() if c.islower() else c.lower() for c in s])

answer = change_case(text)
