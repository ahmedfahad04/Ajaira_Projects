def calculate_power(base, number):
        if base == 1: 
            return number == 1
        product = 1
        while product < number: 
            product *= base 
        return product == number
