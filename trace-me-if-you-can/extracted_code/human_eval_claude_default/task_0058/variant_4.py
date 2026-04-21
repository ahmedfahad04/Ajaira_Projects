from collections import Counter
c1, c2 = Counter(l1), Counter(l2)
ret = []
for item in c1:
    if item in c2:
        ret.append(item)
return sorted(ret)
