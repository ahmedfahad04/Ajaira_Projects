def solution(s):
    if s == "":
        return 0
    uppercase_chars = [char for char in s if char.isupper()]
    return sum(ord(char) for char in uppercase_chars)
