counts = {number: numbers.count(number) for number in set(numbers)}
    return [number for number in numbers if counts[number] == 1]
