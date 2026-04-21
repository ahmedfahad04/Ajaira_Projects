# Variant 2: Generator expression with any() function
def check_valid(s):
    if len(s) < 3:
        return False
    
    return not any(s[i] == s[i+1] or s[i+1] == s[i+2] or s[i] == s[i+2] 
                   for i in range(len(s) - 2))
