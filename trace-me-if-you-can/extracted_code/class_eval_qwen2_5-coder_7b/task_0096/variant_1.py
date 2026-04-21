class WeatherDataHandler:
    def __init__(self, location) -> None:
        self.temp_value = None
        self.condition = None
        self.location = location
        self.location_weather = {}
    
    def fetch_weather(self, weather_data, temp_scale = 'C'):
        self.location_weather = weather_data
        if self.location not in weather_data:
            return False
        else:
            self.temp_value = weather_data[self.location]['temperature']
            self.condition = weather_data[self.location]['weather']
        if weather_data[self.location]['temperature scale'] != temp_scale:
            if temp_scale == 'C':
                return self.convert_to_celsius(), self.condition
            elif temp_scale == 'F':
                return self.convert_to_fahrenheit(), self.condition
        else:
            return self.temp_value, self.condition
    
    def update_location(self, location):
        self.location = location

    def convert_to_celsius(self):
        return (self.temp_value - 32) * 5/9

    def convert_to_fahrenheit(self):
        return (self.temp_value * 9/5) + 32
