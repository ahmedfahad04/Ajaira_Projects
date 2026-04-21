# Variant 4: While loop with manual indexing
count = 0
i = 0
while i < len(s):
    if s[i] in "AEIOU":
        count += 1
    i += 2
return count
