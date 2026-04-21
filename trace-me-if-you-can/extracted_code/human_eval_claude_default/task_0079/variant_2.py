# Version 2: Manual bit manipulation approach
binary_str = ""
temp = decimal
if temp == 0:
    binary_str = "0"
else:
    while temp > 0:
        binary_str = str(temp % 2) + binary_str
        temp //= 2
return "db" + binary_str + "db"
