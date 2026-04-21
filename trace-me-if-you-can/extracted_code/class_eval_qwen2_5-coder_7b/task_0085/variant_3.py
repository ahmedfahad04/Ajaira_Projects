import time

class ClimateSystem:
    def __init__(self, starting_temp, target_temp, mode):
        self.current_temperature = starting_temp
        self.target_temperature = target_temp
        self.mode = mode

    def get_target_temperature(self):
        return self.target_temperature

    def set_target_temperature(self, temperature):
        self.target_temperature = temperature

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        if mode in ['heat', 'cool']:
            self.mode = mode
        else:
            return False

    def auto_decide_mode(self):
        if self.current_temperature < self.target_temperature:
            self.mode = 'heat'
        else:
            self.mode = 'cool'

    def auto_manage_conflict(self):
        if self.current_temperature > self.target_temperature:
            if self.mode == 'cool':
                return True
            else:
                self.auto_decide_mode()
                return False
        else:
            if self.mode == 'heat':
                return True
            else:
                self.auto_decide_mode()
                return False

    def simulate_operation(self):
        self.auto_decide_mode()
        total_time = 0
        if self.mode == 'heat':
            while self.current_temperature < self.target_temperature:
                self.current_temperature += 1
                total_time += 1
        else:
            while self.current_temperature > self.target_temperature:
                self.current_temperature -= 1
                total_time += 1
        return total_time
