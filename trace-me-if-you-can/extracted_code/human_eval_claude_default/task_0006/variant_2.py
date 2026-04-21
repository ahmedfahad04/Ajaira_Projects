def separate_paren_groups(paren_string):
    results = []
    for group in paren_string.split(' '):
        if not group:
            continue
        depth = max_depth = 0
        for char in group:
            if char == '(':
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            else:
                depth -= 1
        results.append(max_depth)
    return results
