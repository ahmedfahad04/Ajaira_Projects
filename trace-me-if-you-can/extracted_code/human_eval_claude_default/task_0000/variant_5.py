i = 0
while i < len(numbers):
    j = i + 1
    while j < len(numbers):
        if abs(numbers[i] - numbers[j]) < threshold:
            return True
        j += 1
    i += 1
return False
