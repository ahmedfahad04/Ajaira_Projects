</response>

<response>
<code>
def parse_groups(paren_string):
    def find_matching_paren(s, start):
        count = 1
        for i in range(start + 1, len(s)):
            if s[i] == '(':
                count += 1
            elif s[i] == ')':
                count -= 1
                if count == 0:
                    return i
        return -1
    
    result = []
    i = 0
    
    while i < len(paren_string):
        if paren_string[i] == '(':
            end = find_matching_paren(paren_string, i)
            result.append(paren_string[i:end+1])
            i = end + 1
        else:
            i += 1
    
    return result
