class WeatherSystem:
    CONVERSION_FUNCTIONS = {
        ('celsius', 'fahrenheit'): lambda temp: (temp * 9/5) + 32,
        ('fahrenheit', 'celsius'): lambda temp: (temp - 32) * 5/9
    }
    
    def __init__(self, city) -> None:
        self.temperature = None
        self.weather = None
        self.city = city
        self.weather_list = {}
    
    def query(self, weather_list, tmp_units='celsius'):
        self.weather_list = weather_list
        
        try:
            city_weather = weather_list[self.city]
        except KeyError:
            return False
        
        self.temperature = city_weather['temperature']
        self.weather = city_weather['weather']
        source_units = city_weather['temperature units']
        
        if source_units != tmp_units:
            conversion_key = (source_units, tmp_units)
            converter = self.CONVERSION_FUNCTIONS.get(conversion_key)
            if converter:
                return converter(self.temperature), self.weather
        
        return self.temperature, self.weather
    
    def set_city(self, city):
        self.city = city

    def celsius_to_fahrenheit(self):
        return (self.temperature * 9/5) + 32

    def fahrenheit_to_celsius(self):
        return (self.temperature - 32) * 5/9
