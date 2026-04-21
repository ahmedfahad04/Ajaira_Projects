from functools import reduce

def process_bracket(acc, bracket):
    depth, valid = acc
    new_depth = depth + (1 if bracket == "<" else -1)
    return (new_depth, valid and new_depth >= 0)

final_depth, is_valid = reduce(process_bracket, brackets, (0, True))
return is_valid and final_depth == 0
