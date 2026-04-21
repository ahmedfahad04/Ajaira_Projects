num_sevens = 0
for i in range(n):
    if i % 11 == 0 or i % 13 == 0:
        num_sevens += sum(1 for digit in str(i) if digit == '7')
return num_sevens
