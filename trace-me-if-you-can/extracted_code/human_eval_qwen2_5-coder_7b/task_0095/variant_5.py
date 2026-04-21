def check_dict_key_case(my_dict):
            if not my_dict:
                return False
            keys = list(my_dict.keys())
            if not all(isinstance(k, str) for k in keys):
                return False
            initial_state = None
            for key in keys:
                if initial_state is None:
                    initial_state = key.isupper() and "upper" or key.islower() and "lower" or "mixed"
                if (initial_state == "upper" and not key.isupper()) or (initial_state == "lower" and not key.islower()):
                    return False
            return initial_state == "upper" or initial_state == "lower"
