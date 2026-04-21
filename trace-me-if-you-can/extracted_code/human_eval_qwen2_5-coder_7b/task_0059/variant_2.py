def identify_max_prime_factor(num):
    max_prime = 1
    test_num = 2
    while test_num * test_num <= num:
        if num % test_num == 0:
            max_prime = test_num
            while num % test_num == 0:
                num //= test_num
        test_num += 1
    if num > 1:
        max_prime = num
    return max_prime
