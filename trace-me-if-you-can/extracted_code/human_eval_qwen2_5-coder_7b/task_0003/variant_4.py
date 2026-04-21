running_total = 0

for operation in ops:
    running_total += operation
    if running_total < 0:
        return True

return False
