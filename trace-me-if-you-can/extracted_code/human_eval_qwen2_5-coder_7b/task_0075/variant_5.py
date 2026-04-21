def prime(n):
    for j in range(2, n):
        if n % j == 0:
            return False
    return True

def triple_product_search(a):
    for i in range(2, 101):
        if prime(i):
            for j in range(i, 101):
                if prime(j):
                    for k in range(j, 101):
                        if prime(k):
                            if i * j * k == a:
                                return True
    return False
