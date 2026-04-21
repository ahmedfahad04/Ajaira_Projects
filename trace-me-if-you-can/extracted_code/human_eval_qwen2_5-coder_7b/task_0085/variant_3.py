result = 0
for index in range(1, len(lst), 2):
    if lst[index] % 2 == 0:
        result += lst[index]
return result
