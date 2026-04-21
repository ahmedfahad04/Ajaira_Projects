if sum(q) > w:
    return False

for idx, val in enumerate(q[:len(q)//2]):
    if val != q[-(idx+1)]:
        return False
return True
