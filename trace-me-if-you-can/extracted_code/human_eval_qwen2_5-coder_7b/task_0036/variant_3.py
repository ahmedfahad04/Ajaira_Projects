total_sevens = 0
for potential_seven in range(n):
    if potential_seven % 11 == 0 or potential_seven % 13 == 0:
        total_sevens += str(potential_seven).count('7')
return total_sevens
