# Variant 4: While loop with index tracking
def check_valid(s):
    if len(s) < 3:
        return False
    
    i = 0
    while i <= len(s) - 3:
        current, next_char, after_next = s[i], s[i+1], s[i+2]
        if current == next_char or next_char == after_next or current == after_next:
            return False
        i += 1
    return True
