import time

class ClimateController:
    def __init__(self, initial_temp, desired_temp, control_mode):
        self.temperature = initial_temp
        self.desired_temp = desired_temp
        self.control_mode = control_mode

    def get_desired_temp(self):
        return self.desired_temp

    def set_desired_temp(self, temp):
        self.desired_temp = temp

    def get_control_mode(self):
        return self.control_mode

    def set_control_mode(self, mode):
        if mode in ['heat', 'cool']:
            self.control_mode = mode
        else:
            return False

    def auto_select_mode(self):
        if self.temperature < self.desired_temp:
            self.control_mode = 'heat'
        else:
            self.control_mode = 'cool'

    def auto_handle_conflict(self):
        if self.temperature > self.desired_temp:
            if self.control_mode == 'cool':
                return True
            else:
                self.auto_select_mode()
                return False
        else:
            if self.control_mode == 'heat':
                return True
            else:
                self.auto_select_mode()
                return False

    def simulate_operation(self):
        self.auto_select_mode()
        total_time = 0
        if self.control_mode == 'heat':
            while self.temperature < self.desired_temp:
                self.temperature += 1
                total_time += 1
        else:
            while self.temperature > self.desired_temp:
                self.temperature -= 1
                total_time += 1
        return total_time
