unique_numbers = set(numbers)
    return [number for number in numbers if numbers.count(number) == 1]
