import time

class TemperatureManager:
    def __init__(self, current_temp, target_temp, operation_mode):
        self.current_temp = current_temp
        self.target_temp = target_temp
        self.operation_mode = operation_mode

    def get_target_temp(self):
        return self.target_temp

    def set_target_temp(self, temp):
        self.target_temp = temp

    def get_operation_mode(self):
        return self.operation_mode

    def set_operation_mode(self, mode):
        if mode in ['heat', 'cool']:
            self.operation_mode = mode
        else:
            return False

    def auto_determine_mode(self):
        if self.current_temp < self.target_temp:
            self.operation_mode = 'heat'
        else:
            self.operation_mode = 'cool'

    def auto_respond_to_conflict(self):
        if self.current_temp > self.target_temp:
            if self.operation_mode == 'cool':
                return True
            else:
                self.auto_determine_mode()
                return False
        else:
            if self.operation_mode == 'heat':
                return True
            else:
                self.auto_determine_mode()
                return False

    def simulate_operation(self):
        self.auto_determine_mode()
        total_time = 0
        if self.operation_mode == 'heat':
            while self.current_temp < self.target_temp:
                self.current_temp += 1
                total_time += 1
        else:
            while self.current_temp > self.target_temp:
                self.current_temp -= 1
                total_time += 1
        return total_time
