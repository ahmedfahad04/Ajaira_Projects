def has_consistent_case_keys(dict):
    key_list = list(dict.keys())
    
    def is_valid_case_key(key):
        return isinstance(key, str) and (key.isupper() or key.islower())
    
    def all_same_case(keys):
        if not keys:
            return False
        upper_count = sum(1 for k in keys if k.isupper())
        lower_count = sum(1 for k in keys if k.islower())
        return upper_count == len(keys) or lower_count == len(keys)
    
    if len(key_list) == 0:
        return False
    
    valid_keys = [k for k in key_list if is_valid_case_key(k)]
    
    return len(valid_keys) == len(key_list) and all_same_case(valid_keys)
