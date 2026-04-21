def find_median(numbers):
    numbers = sorted(numbers)
    mid = len(numbers) // 2
    if len(numbers) % 2 == 1:
        return numbers[mid]
    else:
        return (numbers[mid - 1] + numbers[mid]) / 2.0
