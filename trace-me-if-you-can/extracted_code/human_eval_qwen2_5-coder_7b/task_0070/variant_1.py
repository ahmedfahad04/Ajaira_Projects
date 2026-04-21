output, toggle = [], True
    while lst:
        value = min(lst) if toggle else max(lst)
        output.append(value)
        lst.remove(value)
        toggle = not toggle
    return output
