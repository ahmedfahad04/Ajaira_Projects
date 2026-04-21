divisor = None
    for index in range(n, 0, -1):
        if n % index == 0:
            divisor = index
            break

    print(divisor)
