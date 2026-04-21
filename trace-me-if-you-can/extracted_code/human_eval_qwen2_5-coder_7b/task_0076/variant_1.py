def is_power_of_n(n, x):
        if n == 1: 
            return x == 1
        power = 1
        while power < x: 
            power *= n 
        return power == x
