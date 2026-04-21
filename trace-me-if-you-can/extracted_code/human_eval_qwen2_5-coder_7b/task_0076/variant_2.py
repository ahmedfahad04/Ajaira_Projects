def check_power_relation(base, target):
        if base == 1: 
            return target == 1
        result = 1
        while result < target: 
            result *= base 
        return result == target
