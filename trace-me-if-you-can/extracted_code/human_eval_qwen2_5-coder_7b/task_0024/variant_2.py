def find_largest_divisor(num):
        for index in reversed(range(1, num + 1)):
            if num % index == 0:
                return index

    n = 100
    largest_divisor = find_largest_divisor(n)
    print(largest_divisor)
