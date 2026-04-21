import time

class Thermostat:
    def __init__(self, current_temperature, target_temperature, mode):
        self.current_temperature = current_temperature
        self.target_temperature = target_temperature
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

    def auto_set_mode(self):
        self.mode = 'heat' if self.current_temperature < self.target_temperature else 'cool'

    def auto_check_conflict(self):
        temperature_delta = self.current_temperature - self.target_temperature
        
        if temperature_delta > 0:  # Current temp is higher
            conflict_exists = self.mode == 'cool'
        else:  # Current temp is lower or equal
            conflict_exists = self.mode == 'heat'
        
        if conflict_exists:
            return True
        
        self.auto_set_mode()
        return False

    def simulate_operation(self):
        self.auto_set_mode()
        
        def calculate_time_to_target():
            if self.mode == 'heat':
                return max(0, self.target_temperature - self.current_temperature)
            else:  # mode == 'cool'
                return max(0, self.current_temperature - self.target_temperature)
        
        time_needed = calculate_time_to_target()
        
        # Update temperature in single operation instead of loop
        self.current_temperature = self.target_temperature
        
        return time_needed
