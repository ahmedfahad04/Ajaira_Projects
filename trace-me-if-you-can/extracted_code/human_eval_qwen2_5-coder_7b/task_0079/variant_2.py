decimal_value = decimal
binary_value = bin(decimal_value)[2:]
return f"db{binary_value}db"
