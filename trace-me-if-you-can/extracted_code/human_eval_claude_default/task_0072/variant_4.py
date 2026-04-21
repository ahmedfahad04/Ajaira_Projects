if sum(q) > w:
    return False

n = len(q)
return all(q[k] == q[n-1-k] for k in range(n//2))
