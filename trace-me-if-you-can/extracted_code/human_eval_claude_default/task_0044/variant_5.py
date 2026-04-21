stack = []
temp_x = x
while temp_x > 0:
    stack.append(str(temp_x % base))
    temp_x //= base

ret = ""
while stack:
    ret += stack.pop()
return ret
