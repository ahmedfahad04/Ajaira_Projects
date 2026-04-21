# Version 5: Using join with bit extraction
bits = []
temp = decimal
if temp == 0:
    bits = ['0']
else:
    while temp:
        bits.append(str(temp & 1))
        temp >>= 1
    bits.reverse()
return "db" + "".join(bits) + "db"
