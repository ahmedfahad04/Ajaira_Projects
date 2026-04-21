def count_chars(s, index=0):
    try:
        s[index]
        return 1 + count_chars(s, index + 1)
    except IndexError:
        return 0
return count_chars(string)
