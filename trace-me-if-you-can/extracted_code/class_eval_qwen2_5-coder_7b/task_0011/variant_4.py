class BitwiseOperations:
    @staticmethod
    def add_flag(state, flag):
        BitwiseOperations.check_values(state, flag)
        return state | flag

    @staticmethod
    def contains_flag(state, flag):
        BitwiseOperations.check_values(state, flag)
        return (state & flag) == flag

    @staticmethod
    def remove_flag(state, flag):
        BitwiseOperations.check_values(state, flag)
        if BitwiseOperations.contains_flag(state, flag):
            return state ^ flag
        return state

    @staticmethod
    def check_values(*values):
        for value in values:
            if value < 0:
                raise ValueError(f"{value} must be non-negative")
            if value % 2 != 0:
                raise ValueError(f"{value} must be even")
