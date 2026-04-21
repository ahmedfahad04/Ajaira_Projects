highest = None
res = []

for num in numbers:
    if highest is None or num > highest:
        highest = num
    res.append(highest)

return res
