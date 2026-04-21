def parse_parentheses(paren_string):
    import re
    
    def find_balanced_groups(s):
        groups = []
        i = 0
        while i < len(s):
            if s[i] == '(':
                count = 0
                start = i
                for j in range(i, len(s)):
                    if s[j] == '(':
                        count += 1
                    elif s[j] == ')':
                        count -= 1
                    if count == 0:
                        groups.append(s[start:j+1])
                        i = j + 1
                        break
            else:
                i += 1
        return groups
    
    return find_balanced_groups(paren_string)
