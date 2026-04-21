class BitStatusUtil:
    @staticmethod
    def add(states, stat):
        BitStatusUtil._validate_inputs(states, stat)
        return states | stat

    @staticmethod
    def has(states, stat):
        BitStatusUtil._validate_inputs(states, stat)
        return bool(states & stat == stat)

    @staticmethod
    def remove(states, stat):
        BitStatusUtil._validate_inputs(states, stat)
        return states ^ stat if (states & stat) == stat else states

    @staticmethod
    def _validate_inputs(states, stat):
        for value in [states, stat]:
            if value < 0:
                raise ValueError(f"{value} must be greater than or equal to 0")
            if value & 1:
                raise ValueError(f"{value} not even")
