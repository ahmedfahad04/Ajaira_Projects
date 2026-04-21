def find_max_prime(lst):
    max_prime = 0
    for number in lst:
        if number > max_prime and all(number % i != 0 for i in range(2, int(number**0.5) + 1)):
            max_prime = number
    return sum(int(digit) for digit in str(max_prime))

# Example usage:
result = find_max_prime([10, 15, 7, 11, 4])
print(result)  # Output will be the sum of digits of the largest prime number in the list
