class BitStatusUtil:
    @staticmethod
    def add(states, stat):
        BitStatusUtil.check([states, stat])
        return states | stat

    @staticmethod
    def has(states, stat):
        BitStatusUtil.check([states, stat])
        intersection = states & stat
        return intersection == stat

    @staticmethod
    def remove(states, stat):
        BitStatusUtil.check([states, stat])
        contains_stat = BitStatusUtil.has(states, stat)
        if contains_stat:
            return states ^ stat
        else:
            return states

    @staticmethod
    def check(args):
        validation_errors = []
        for arg in args:
            if arg < 0:
                validation_errors.append(ValueError(f"{arg} must be greater than or equal to 0"))
            elif arg % 2 != 0:
                validation_errors.append(ValueError(f"{arg} not even"))
        
        if validation_errors:
            raise validation_errors[0]
