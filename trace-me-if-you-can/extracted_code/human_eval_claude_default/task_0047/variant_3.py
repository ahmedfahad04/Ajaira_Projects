sorted_l = sorted(l)
n = len(sorted_l)
return (lambda lst, size: lst[size // 2] if size % 2 == 1 else (lst[size // 2 - 1] + lst[size // 2]) / 2.0)(sorted_l, n)
