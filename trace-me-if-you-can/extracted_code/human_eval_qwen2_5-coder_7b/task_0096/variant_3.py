def get_primes(n):
    primes_list = []
    for number in range(2, n):
        if number > 1:
            for i in range(2, number):
                if (number % i) == 0:
                    break
            else:
                primes_list.append(number)
    return primes_list
