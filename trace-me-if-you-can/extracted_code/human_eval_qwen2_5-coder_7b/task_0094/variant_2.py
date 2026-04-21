def prime_finder(lst):
    largest_prime = 0
    for number in lst:
        if number > largest_prime and not any(number % i == 0 for i in range(2, int(number**0.5) + 1)):
            largest_prime = number
    return sum(int(digit) for digit in str(largest_prime))

# Example usage:
result = prime_finder([12, 19, 7, 13, 5])
print(result)  # Output will be the sum of digits of the largest prime number in the list
