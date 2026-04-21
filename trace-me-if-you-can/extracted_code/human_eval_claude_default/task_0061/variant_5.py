from functools import reduce

def accumulate_depth(acc, bracket):
    depth, valid = acc
    new_depth = depth + (1 if bracket == "(" else -1)
    return (new_depth, valid and new_depth >= 0)

final_depth, is_valid = reduce(accumulate_depth, brackets, (0, True))
return is_valid and final_depth == 0
