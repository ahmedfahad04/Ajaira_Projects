def parse_parentheses(paren_string):
    def extract_groups(s, pos=0, groups=None):
        if groups is None:
            groups = []
        
        while pos < len(s):
            if s[pos] == '(':
                depth = 1
                start = pos
                pos += 1
                
                while pos < len(s) and depth > 0:
                    if s[pos] == '(':
                        depth += 1
                    elif s[pos] == ')':
                        depth -= 1
                    pos += 1
                
                if depth == 0:
                    groups.append(s[start:pos])
            else:
                pos += 1
        
        return groups
    
    return extract_groups(paren_string)
