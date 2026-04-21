import time
from dataclasses import dataclass
from typing import Literal

@dataclass
class ThermostatState:
    current_temperature: float
    target_temperature: float
    mode: Literal['heat', 'cool']

class Thermostat:
    def __init__(self, current_temperature, target_temperature, mode):
        self._state = ThermostatState(current_temperature, target_temperature, mode)

    def get_target_temperature(self):
        return self._state.target_temperature

    def set_target_temperature(self, temperature):
        self._state.target_temperature = temperature

    def get_mode(self):
        return self._state.mode

    def set_mode(self, mode):
        if mode in ['heat', 'cool']:
            self._state.mode = mode
        else:
            return False

    def auto_set_mode(self):
        self._state.mode = 'heat' if self._state.current_temperature < self._state.target_temperature else 'cool'

    def auto_check_conflict(self):
        temp_diff = self._state.current_temperature - self._state.target_temperature
        expected_mode = 'cool' if temp_diff > 0 else 'heat'
        
        if self._state.mode == expected_mode:
            return True
        else:
            self.auto_set_mode()
            return False

    def simulate_operation(self):
        self.auto_set_mode()
        temp_delta = 1 if self._state.mode == 'heat' else -1
        use_time = abs(self._state.target_temperature - self._state.current_temperature)
        self._state.current_temperature = self._state.target_temperature
        return int(use_time)
