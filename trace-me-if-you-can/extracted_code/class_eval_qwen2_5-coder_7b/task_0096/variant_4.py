class AtmosphericConditions:
    def __init__(self, area) -> None:
        self.temperature = None
        self.weather_description = None
        self.area_name = area
        self.area_weather = {}
    
    def request_weather(self, weather_data, temp_unit = 'celsius'):
        self.area_weather = weather_data
        if self.area_name not in weather_data:
            return False
        else:
            self.temperature = weather_data[self.area_name]['temperature']
            self.weather_description = weather_data[self.area_name]['weather']
        if weather_data[self.area_name]['temperature unit'] != temp_unit:
            if temp_unit == 'celsius':
                return self.temperature_to_celsius(), self.weather_description
            elif temp_unit == 'fahrenheit':
                return self.convert_to_fahrenheit(), self.weather_description
        else:
            return self.temperature, self.weather_description
    
    def update_area(self, area):
        self.area_name = area

    def temperature_to_celsius(self):
        return (self.temperature - 32) * 5/9

    def convert_to_fahrenheit(self):
        return (self.temperature * 9/5) + 32
