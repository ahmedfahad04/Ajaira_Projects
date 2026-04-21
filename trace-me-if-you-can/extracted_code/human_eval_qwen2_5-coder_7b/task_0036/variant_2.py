sevens_count = 0
for num in range(n):
    if num % 11 == 0 or num % 13 == 0:
        sevens_count += str(num).count('7')
return sevens_count
