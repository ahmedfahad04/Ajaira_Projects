class BitStatusUtil:
    @staticmethod
    def add(states, stat):
        return BitStatusUtil._execute_operation(states, stat, lambda s, t: s | t)

    @staticmethod
    def has(states, stat):
        return BitStatusUtil._execute_operation(states, stat, lambda s, t: (s & t) == t)

    @staticmethod
    def remove(states, stat):
        def removal_logic(s, t):
            return s ^ t if (s & t) == t else s
        return BitStatusUtil._execute_operation(states, stat, removal_logic)

    @staticmethod
    def _execute_operation(states, stat, operation):
        BitStatusUtil.check([states, stat])
        return operation(states, stat)

    @staticmethod
    def check(args):
        for arg in args:
            if arg < 0:
                raise ValueError(f"{arg} must be greater than or equal to 0")
            if arg % 2 != 0:
                raise ValueError(f"{arg} not even")
