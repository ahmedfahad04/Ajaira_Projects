def calculate_max_prime(lst):
    max_prime = 0
    for value in lst:
        if value > max_prime and not any(value % i == 0 for i in range(2, int(value**0.5) + 1)):
            max_prime = value
    return sum(int(digit) for digit in str(max_prime))

# Example usage:
result = calculate_max_prime([13, 20, 7, 11, 1])
print(result)  # Output will be the sum of digits of the largest prime number in the list
