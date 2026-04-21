result, flag = [], True
    while lst:
        current = min(lst) if flag else max(lst)
        result.append(current)
        lst.remove(current)
        flag = not flag
    return result
