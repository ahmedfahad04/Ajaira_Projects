counts = {number: numbers.count(number) for number in numbers}
    return list(filter(lambda x: counts[x] == 1, numbers))
