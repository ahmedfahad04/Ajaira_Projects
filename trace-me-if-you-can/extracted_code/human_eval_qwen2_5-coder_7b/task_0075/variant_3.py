def is_number_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def find_triplet_product(a):
    for i in range(2, 101):
        if is_number_prime(i):
            for j in range(i, 101):
                if is_number_prime(j):
                    for k in range(j, 101):
                        if is_number_prime(k):
                            if i * j * k == a:
                                return True
    return False
