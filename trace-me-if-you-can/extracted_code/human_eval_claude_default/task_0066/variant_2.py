def solution(s):
    return 0 if s == "" else sum(map(lambda c: ord(c) if c.isupper() else 0, s))
