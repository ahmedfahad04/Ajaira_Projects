unique_s0 = set(s0)
unique_s1 = set(s1)
return unique_s0.issubset(unique_s1) and unique_s1.issubset(unique_s0)
