def check_valid(s):
    if len(s) < 3:
        return False
    
    for i in range(len(s) - 2):
        triplet = [s[i], s[i+1], s[i+2]]
        if len(set(triplet)) != 3:
            return False
    return True
