open_count = 0
for bracket in brackets:
    open_count += 1 if bracket == "<" else -1
    if open_count < 0:
        return False
return open_count == 0
