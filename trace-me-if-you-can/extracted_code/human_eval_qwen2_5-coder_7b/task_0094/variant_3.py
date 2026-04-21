def get_max_prime(lst):
    max_prime = 0
    for num in lst:
        if num > max_prime and not any(num % i == 0 for i in range(2, int(num**0.5) + 1)):
            max_prime = num
    return sum(int(digit) for digit in str(max_prime))

# Example usage:
result = get_max_prime([11, 18, 7, 15, 3])
print(result)  # Output will be the sum of digits of the largest prime number in the list
