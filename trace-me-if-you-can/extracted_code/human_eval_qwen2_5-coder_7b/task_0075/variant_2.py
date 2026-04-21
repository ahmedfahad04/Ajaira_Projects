def prime_check(value):
    for factor in range(2, value):
        if value % factor == 0:
            return False
    return True

def search_triplet_product(search_value):
    for first_value in range(2, 101):
        if prime_check(first_value):
            for second_value in range(first_value, 101):
                if prime_check(second_value):
                    for third_value in range(second_value, 101):
                        if prime_check(third_value):
                            if first_value * second_value * third_value == search_value:
                                return True
    return False
