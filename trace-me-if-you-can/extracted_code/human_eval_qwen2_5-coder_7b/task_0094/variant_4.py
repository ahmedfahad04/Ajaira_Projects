def extract_max_prime(lst):
    max_prime = 0
    for element in lst:
        if element > max_prime and all(element % i != 0 for i in range(2, int(element**0.5) + 1)):
            max_prime = element
    return sum(int(digit) for digit in str(max_prime))

# Example usage:
result = extract_max_prime([14, 17, 7, 19, 2])
print(result)  # Output will be the sum of digits of the largest prime number in the list
