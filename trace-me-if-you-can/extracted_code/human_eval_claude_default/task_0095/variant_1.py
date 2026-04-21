def check_keys_case_consistency(dict):
    if not dict:
        return False
    
    keys = list(dict.keys())
    string_keys = [k for k in keys if isinstance(k, str)]
    
    if len(string_keys) != len(keys):
        return False
    
    if not string_keys:
        return False
    
    all_upper = all(k.isupper() for k in string_keys)
    all_lower = all(k.islower() for k in string_keys)
    
    return all_upper or all_lower
