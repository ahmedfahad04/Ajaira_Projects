if sum(q) > w:
    return False

q_str = ''.join(map(str, q))
return q_str == q_str[::-1]
