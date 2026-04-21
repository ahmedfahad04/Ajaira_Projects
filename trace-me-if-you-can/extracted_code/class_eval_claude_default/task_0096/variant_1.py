class WeatherSystem:
    def __init__(self, city) -> None:
        self.city = city
        self._current_data = {}
    
    def query(self, weather_list, tmp_units='celsius'):
        if self.city not in weather_list:
            return False
        
        city_data = weather_list[self.city]
        self._current_data = city_data
        
        temp = city_data['temperature']
        weather = city_data['weather']
        current_units = city_data['temperature units']
        
        if current_units == tmp_units:
            return temp, weather
        
        converted_temp = self._convert_temperature(temp, current_units, tmp_units)
        return converted_temp, weather
    
    def set_city(self, city):
        self.city = city
    
    def _convert_temperature(self, temp, from_units, to_units):
        if from_units == 'celsius' and to_units == 'fahrenheit':
            return (temp * 9/5) + 32
        elif from_units == 'fahrenheit' and to_units == 'celsius':
            return (temp - 32) * 5/9
        return temp
    
    @property
    def temperature(self):
        return self._current_data.get('temperature')
    
    @property
    def weather(self):
        return self._current_data.get('weather')
    
    def celsius_to_fahrenheit(self):
        return (self.temperature * 9/5) + 32

    def fahrenheit_to_celsius(self):
        return (self.temperature - 32) * 5/9
