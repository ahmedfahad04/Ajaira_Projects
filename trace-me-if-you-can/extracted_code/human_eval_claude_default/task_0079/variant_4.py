# Version 4: Recursive approach for binary conversion
def to_binary(n):
    if n == 0:
        return "0"
    elif n == 1:
        return "1"
    else:
        return to_binary(n // 2) + str(n % 2)

return "db" + to_binary(decimal) + "db"
