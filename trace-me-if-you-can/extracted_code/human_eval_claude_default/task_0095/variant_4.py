def check_dictionary_case_uniformity(dict):
    keys = dict.keys()
    
    if len(keys) == 0:
        return False
    
    case_checker = lambda k: 'upper' if k.isupper() else ('lower' if k.islower() else 'mixed')
    
    cases = []
    for key in keys:
        if not isinstance(key, str):
            return False
        case_type = case_checker(key)
        if case_type == 'mixed':
            return False
        cases.append(case_type)
    
    return len(set(cases)) == 1
