class BitmaskHandler:
    @staticmethod
    def add_state(current_state, new_state):
        BitmaskHandler.check_values([current_state, new_state])
        return current_state | new_state

    @staticmethod
    def check_state(current_state, check_state):
        BitmaskHandler.check_values([current_state, check_state])
        return (current_state & check_state) == check_state

    @staticmethod
    def remove_state(current_state, state_to_remove):
        BitmaskHandler.check_values([current_state, state_to_remove])
        if BitmaskHandler.check_state(current_state, state_to_remove):
            return current_state ^ state_to_remove
        return current_state

    @staticmethod
    def check_values(*values):
        for value in values:
            if value < 0:
                raise ValueError(f"{value} must be non-negative")
            if value % 2 != 0:
                raise ValueError(f"{value} must be even")
