def check_dict_state(input_dict):
            if not input_dict:
                return False
            keys = list(input_dict.keys())
            if not all(isinstance(k, str) for k in keys):
                return False
            if not keys:
                return True
            initial_state = keys[0].isupper() and "upper" or keys[0].islower() and "lower" or "mixed"
            for key in keys[1:]:
                if (initial_state == "upper" and not key.isupper()) or (initial_state == "lower" and not key.islower()):
                    return False
            return initial_state == "upper" or initial_state == "lower"
