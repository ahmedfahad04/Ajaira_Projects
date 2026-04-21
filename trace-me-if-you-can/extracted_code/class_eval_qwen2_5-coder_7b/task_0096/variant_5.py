class ClimaticConditions:
    def __init__(self, location) -> None:
        self.temp = None
        self.weather_state = None
        self.location_name = location
        self.location_weather = {}
    
    def fetch_weather(self, weather_info, temp_scale = 'C'):
        self.location_weather = weather_info
        if self.location_name not in weather_info:
            return False
        else:
            self.temp = weather_info[self.location_name]['temperature']
            self.weather_state = weather_info[self.location_name]['weather']
        if weather_info[self.location_name]['temperature unit'] != temp_scale:
            if temp_scale == 'C':
                return self.temp_to_celsius(), self.weather_state
            elif temp_scale == 'F':
                return self.celsius_to_fahrenheit(), self.weather_state
        else:
            return self.temp, self.weather_state
    
    def relocate(self, location):
        self.location_name = location

    def celsius_to_fahrenheit(self):
        return (self.temp * 9/5) + 32

    def temp_to_celsius(self):
        return (self.temp - 32) * 5/9
