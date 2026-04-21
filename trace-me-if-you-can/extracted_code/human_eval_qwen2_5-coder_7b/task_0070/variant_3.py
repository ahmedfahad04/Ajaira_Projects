outcomes, active = [], True
    while lst:
        value = min(lst) if active else max(lst)
        outcomes.append(value)
        lst.remove(value)
        active = not active
    return outcomes
