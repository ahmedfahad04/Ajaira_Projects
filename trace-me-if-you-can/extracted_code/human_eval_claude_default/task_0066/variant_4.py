def solution(s):
    return sum(ord(c) for c in s if c.isupper()) if s else 0
