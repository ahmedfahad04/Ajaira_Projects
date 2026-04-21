def verify_exponentiation(base, exponent):
        if base == 1: 
            return exponent == 1
        current_power = 1
        while current_power < exponent: 
            current_power *= base 
        return current_power == exponent
