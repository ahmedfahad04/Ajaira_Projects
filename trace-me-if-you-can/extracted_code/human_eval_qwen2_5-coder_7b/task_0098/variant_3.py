count = 0
    for idx, char in enumerate(s, start=0):
        if idx % 2 == 0 and char in "AEIOU":
            count += 1
    return count
