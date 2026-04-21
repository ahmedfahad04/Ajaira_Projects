open_brackets = brackets.count("<")
close_brackets = brackets.count(">")

if open_brackets != close_brackets:
    return False

balance = 0
for char in brackets:
    balance += 1 if char == "<" else -1
    if balance < 0:
        return False
return True
