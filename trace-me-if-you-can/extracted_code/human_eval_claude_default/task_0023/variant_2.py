# Variant 2: Manual counting with while loop
count = 0
index = 0
try:
    while True:
        string[index]
        count += 1
        index += 1
except IndexError:
    pass
return count
