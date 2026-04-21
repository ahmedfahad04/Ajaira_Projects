def get_highest_divisor(value):
        for potential_divisor in range(value, 0, -1):
            if value % potential_divisor == 0:
                return potential_divisor

    n = 100
    highest_divisor = get_highest_divisor(n)
    print(highest_divisor)
