def find_nearby(numbers, threshold):
    for i, num in enumerate(numbers):
        for j, num2 in enumerate(numbers):
            if i != j:
                dist = abs(num - num2)
                if dist < threshold:
                    return True
    return False
