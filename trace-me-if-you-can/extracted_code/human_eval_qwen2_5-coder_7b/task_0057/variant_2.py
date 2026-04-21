def is_sequence_ordered(lst):
    return lst == sorted(lst) or lst == sorted(lst, reverse=True)
