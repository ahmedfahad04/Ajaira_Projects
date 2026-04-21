def check_prime(number):
    for divisor in range(2, number):
        if number % divisor == 0:
            return False
    return True

def find_product_triplet(target_product):
    for potential_i in range(2, 101):
        if check_prime(potential_i):
            for potential_j in range(potential_i, 101):
                if check_prime(potential_j):
                    for potential_k in range(potential_j, 101):
                        if check_prime(potential_k):
                            if potential_i * potential_j * potential_k == target_product:
                                return True
    return False
