def prime_checker(input_str):
    str_len = len(input_str)
    if str_len == 0 or str_len == 1:
        return False
    for i in range(2, str_len):
        if str_len % i == 0:
            return False
    return True
