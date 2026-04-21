def validate_dictionary_key_case(dict):
    keys = dict.keys()
    
    try:
        if len(keys) == 0:
            return False
        
        first_key = next(iter(keys))
        if not isinstance(first_key, str):
            return False
        
        target_case = first_key.isupper() if first_key.isupper() else (first_key.islower() if first_key.islower() else None)
        if target_case is None:
            return False
        
        for key in keys:
            if not isinstance(key, str):
                return False
            if target_case and not key.isupper():
                return False
            if not target_case and not key.islower():
                return False
        
        return True
    except StopIteration:
        return False
