def check_valid(s):
    if len(s) < 3:
        return True
    
    def has_duplicate_in_triplet(string, start_idx):
        if start_idx > len(string) - 3:
            return False
        
        triplet = string[start_idx:start_idx + 3]
        if triplet[0] == triplet[1] or triplet[1] == triplet[2] or triplet[0] == triplet[2]:
            return True
        
        return has_duplicate_in_triplet(string, start_idx + 1)
    
    return not has_duplicate_in_triplet(s, 0)
