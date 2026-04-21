def check_prime_number(n):
    for divisor in range(2, n):
        if n % divisor == 0:
            return False
    return True

def search_for_triplet_product(target_product):
    for potential_i in range(2, 101):
        if check_prime_number(potential_i):
            for potential_j in range(potential_i, 101):
                if check_prime_number(potential_j):
                    for potential_k in range(potential_j, 101):
                        if check_prime_number(potential_k):
                            if potential_i * potential_j * potential_k == target_product:
                                return True
    return False
