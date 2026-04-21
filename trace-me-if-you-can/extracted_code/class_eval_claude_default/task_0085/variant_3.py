import time
from enum import Enum

class Mode(Enum):
    HEAT = 'heat'
    COOL = 'cool'

class Thermostat:
    def __init__(self, current_temperature, target_temperature, mode):
        self.current_temperature = current_temperature
        self.target_temperature = target_temperature
        self.mode = mode
        self._mode_handlers = {
            'heat': self._handle_heating,
            'cool': self._handle_cooling
        }

    def get_target_temperature(self):
        return self.target_temperature

    def set_target_temperature(self, temperature):
        self.target_temperature = temperature

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        valid_modes = [m.value for m in Mode]
        if mode not in valid_modes:
            return False
        self.mode = mode

    def auto_set_mode(self):
        self.mode = Mode.HEAT.value if self._needs_heating() else Mode.COOL.value

    def _needs_heating(self):
        return self.current_temperature < self.target_temperature

    def _needs_cooling(self):
        return self.current_temperature > self.target_temperature

    def auto_check_conflict(self):
        conflict_conditions = [
            (self._needs_cooling() and self.mode == Mode.COOL.value),
            (self._needs_heating() and self.mode == Mode.HEAT.value)
        ]
        
        has_conflict = any(conflict_conditions)
        if not has_conflict:
            self.auto_set_mode()
        
        return has_conflict

    def simulate_operation(self):
        self.auto_set_mode()
        return self._mode_handlers[self.mode]()

    def _handle_heating(self):
        time_units = 0
        while self.current_temperature < self.target_temperature:
            self.current_temperature += 1
            time_units += 1
        return time_units

    def _handle_cooling(self):
        time_units = 0
        while self.current_temperature > self.target_temperature:
            self.current_temperature -= 1
            time_units += 1
        return time_units
