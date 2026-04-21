output_list, is_max_turn = [], True
    while lst:
        choice = max(lst) if is_max_turn else min(lst)
        output_list.append(choice)
        lst.remove(choice)
        is_max_turn = not is_max_turn
    return output_list
