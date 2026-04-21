class WeatherSystem:
    def __init__(self, city) -> None:
        self.temperature = None
        self.weather = None
        self.city = city
        self.weather_list = {}
    
    def query(self, weather_list, tmp_units='celsius'):
        self.weather_list = weather_list
        
        if not self._city_exists_in_weather_data():
            return False
        
        self._load_city_weather_data()
        return self._get_weather_with_converted_temperature(tmp_units)
    
    def _city_exists_in_weather_data(self):
        return self.city in self.weather_list
    
    def _load_city_weather_data(self):
        city_data = self.weather_list[self.city]
        self.temperature = city_data['temperature']
        self.weather = city_data['weather']
    
    def _get_weather_with_converted_temperature(self, target_units):
        source_units = self.weather_list[self.city]['temperature units']
        
        if source_units == target_units:
            return self.temperature, self.weather
        
        converted_temp = self._perform_temperature_conversion(target_units)
        return converted_temp, self.weather
    
    def _perform_temperature_conversion(self, target_units):
        conversion_methods = {
            'celsius': self.fahrenheit_to_celsius,
            'fahrenheit': self.celsius_to_fahrenheit
        }
        
        converter = conversion_methods.get(target_units)
        return converter() if converter else self.temperature
    
    def set_city(self, city):
        self.city = city

    def celsius_to_fahrenheit(self):
        return (self.temperature * 9/5) + 32

    def fahrenheit_to_celsius(self):
        return (self.temperature - 32) * 5/9
