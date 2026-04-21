def is_uniform_case(dictionary):
            if not dictionary:
                return False
            first_key = next(iter(dictionary))
            if not isinstance(first_key, str):
                return False
            case_type = first_key.isupper() and "upper" or first_key.islower() and "lower" or "mixed"
            for key in dictionary:
                if (case_type == "upper" and not key.isupper()) or (case_type == "lower" and not key.islower()):
                    return False
            return case_type == "upper" or case_type == "lower"
