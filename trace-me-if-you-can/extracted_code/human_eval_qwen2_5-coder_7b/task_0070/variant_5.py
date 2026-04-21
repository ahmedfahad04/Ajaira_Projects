result, is_min_turn = [], True
    while lst:
        pick = min(lst) if is_min_turn else max(lst)
        result.append(pick)
        lst.remove(pick)
        is_min_turn = not is_min_turn
    return result
