result = ""
for ch in s:
    shifted_value = (ord(ch) - ord("a") - 5) % 26
    result += chr(shifted_value + ord("a"))
return result
