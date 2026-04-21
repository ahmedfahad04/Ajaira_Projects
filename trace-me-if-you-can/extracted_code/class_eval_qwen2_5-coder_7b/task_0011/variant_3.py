class BitFlagOperations:
    @staticmethod
    def insert(state, flag):
        BitFlagOperations.validate_values(state, flag)
        return state | flag

    @staticmethod
    def check_insert(state, flag):
        BitFlagOperations.validate_values(state, flag)
        return (state & flag) == flag

    @staticmethod
    def remove_insert(state, flag):
        BitFlagOperations.validate_values(state, flag)
        if BitFlagOperations.check_insert(state, flag):
            return state ^ flag
        return state

    @staticmethod
    def validate_values(*values):
        for value in values:
            if value < 0:
                raise ValueError(f"{value} must be greater than or equal to 0")
            if value % 2 != 0:
                raise ValueError(f"{value} must be even")
