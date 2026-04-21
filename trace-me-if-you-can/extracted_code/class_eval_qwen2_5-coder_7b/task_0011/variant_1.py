class BitwiseManager:
    @staticmethod
    def include(state, flag):
        BitwiseManager.validate(state, flag)
        return state | flag

    @staticmethod
    def contains(state, flag):
        BitwiseManager.validate(state, flag)
        return (state & flag) == flag

    @staticmethod
    def exclude(state, flag):
        BitwiseManager.validate(state, flag)
        if BitwiseManager.contains(state, flag):
            return state ^ flag
        return state

    @staticmethod
    def validate(*args):
        for arg in args:
            if arg < 0:
                raise ValueError(f"{arg} must be non-negative")
            if arg % 2 != 0:
                raise ValueError(f"{arg} must be even")
