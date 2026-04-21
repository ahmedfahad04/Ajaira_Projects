def verify_key_case_consistency(dict):
    import itertools
    
    if not dict.keys():
        return False
    
    key_iterator = iter(dict.keys())
    
    # Check first key and determine expected case
    try:
        first_key = next(key_iterator)
        if not isinstance(first_key, str):
            return False
        
        expected_checker = str.isupper if first_key.isupper() else (str.islower if first_key.islower() else None)
        if expected_checker is None:
            return False
        
        # Check remaining keys
        return all(isinstance(key, str) and expected_checker(key) 
                  for key in itertools.chain([first_key], key_iterator))
    
    except StopIteration:
        return False
