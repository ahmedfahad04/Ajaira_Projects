class ClimaticData:
    def __init__(self, region) -> None:
        self.temp = None
        self.weather_state = None
        self.region_name = region
        self.region_weather = {}
    
    def get_weather_info(self, weather_info, temp_unit = 'celsius'):
        self.region_weather = weather_info
        if self.region_name not in weather_info:
            return False
        else:
            self.temp = weather_info[self.region_name]['temperature']
            self.weather_state = weather_info[self.region_name]['weather']
        if weather_info[self.region_name]['temperature unit'] != temp_unit:
            if temp_unit == 'celsius':
                return self.temp_to_celsius(), self.weather_state
            elif temp_unit == 'fahrenheit':
                return self.celsius_to_fahrenheit(), self.weather_state
        else:
            return self.temp, self.weather_state
    
    def change_region(self, region):
        self.region_name = region

    def celsius_to_fahrenheit(self):
        return (self.temp * 9/5) + 32

    def temp_to_celsius(self):
        return (self.temp - 32) * 5/9
