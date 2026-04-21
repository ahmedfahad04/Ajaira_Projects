s = str(x)
n = len(s)
if shift >= n:
    result = ""
    for i in range(n-1, -1, -1):
        result += s[i]
    return result
else:
    result = ""
    for i in range(n - shift, n):
        result += s[i]
    for i in range(0, n - shift):
        result += s[i]
    return result
