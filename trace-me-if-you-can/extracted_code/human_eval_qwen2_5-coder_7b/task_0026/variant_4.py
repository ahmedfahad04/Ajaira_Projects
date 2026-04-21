unique_numbers = set(numbers)
    result = []
    for number in numbers:
        if number in unique_numbers:
            result.append(number)
            unique_numbers.remove(number)
    return result
