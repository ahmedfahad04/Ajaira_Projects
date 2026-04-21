count_of_sevens = 0
for candidate in range(n):
    if candidate % 11 == 0 or candidate % 13 == 0:
        count_of_sevens += str(candidate).count('7')
return count_of_sevens
