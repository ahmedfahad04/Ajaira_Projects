def is_exponential_relationship(base, value):
        if base == 1: 
            return value == 1
        multiplier = 1
        while multiplier < value: 
            multiplier *= base 
        return multiplier == value
