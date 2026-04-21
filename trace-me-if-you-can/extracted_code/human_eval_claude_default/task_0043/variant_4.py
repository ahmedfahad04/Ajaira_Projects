count = {}
for num in l:
    complement = -num
    if complement in count and (complement != num or count[complement] > 1):
        return True
    count[num] = count.get(num, 0) + 1
return False
