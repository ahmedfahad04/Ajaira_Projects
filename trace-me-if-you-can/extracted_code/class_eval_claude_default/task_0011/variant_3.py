class BitStatusUtil:
    @classmethod
    def add(cls, states, stat):
        cls.check([states, stat])
        return states | stat

    @classmethod
    def has(cls, states, stat):
        cls.check([states, stat])
        return (states & stat) == stat

    @classmethod
    def remove(cls, states, stat):
        cls.check([states, stat])
        masked_result = states & stat
        return states ^ stat if masked_result == stat else states

    @classmethod
    def check(cls, args):
        if any(arg < 0 for arg in args):
            negative_args = [arg for arg in args if arg < 0]
            raise ValueError(f"{negative_args[0]} must be greater than or equal to 0")
        if any(arg % 2 for arg in args):
            odd_args = [arg for arg in args if arg % 2]
            raise ValueError(f"{odd_args[0]} not even")
