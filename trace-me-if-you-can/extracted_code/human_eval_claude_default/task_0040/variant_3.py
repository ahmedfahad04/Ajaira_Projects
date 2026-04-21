for i, first in enumerate(l):
    for j, second in enumerate(l[i+1:], i+1):
        for third in l[j+1:]:
            if first + second + third == 0:
                return True
return False
