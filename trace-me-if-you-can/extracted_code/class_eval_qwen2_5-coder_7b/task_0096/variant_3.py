class WeatherStation:
    def __init__(self, place) -> None:
        self.temp = None
        self.condition = None
        self.place_name = place
        self.weather_details = {}
    
    def obtain_weather(self, weather_details, temp_scale = 'C'):
        self.weather_details = weather_details
        if self.place_name not in weather_details:
            return False
        else:
            self.temp = weather_details[self.place_name]['temperature']
            self.condition = weather_details[self.place_name]['weather']
        if weather_details[self.place_name]['temperature scale'] != temp_scale:
            if temp_scale == 'C':
                return self.temp_to_celsius(), self.condition
            elif temp_scale == 'F':
                return self.convert_to_fahrenheit(), self.condition
        else:
            return self.temp, self.condition
    
    def relocate(self, place):
        self.place_name = place

    def convert_to_fahrenheit(self):
        return (self.temp * 9/5) + 32

    def temp_to_celsius(self):
        return (self.temp - 32) * 5/9
