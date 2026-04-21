class TemperatureConverter:
    @staticmethod
    def convert(temperature, from_unit, to_unit):
        if from_unit == to_unit:
            return temperature
        elif from_unit == 'celsius' and to_unit == 'fahrenheit':
            return (temperature * 9/5) + 32
        elif from_unit == 'fahrenheit' and to_unit == 'celsius':
            return (temperature - 32) * 5/9
        return temperature

class WeatherSystem:
    def __init__(self, city) -> None:
        self.temperature = None
        self.weather = None
        self.city = city
        self.weather_list = {}
        self.converter = TemperatureConverter()
    
    def query(self, weather_list, tmp_units='celsius'):
        self.weather_list = weather_list
        
        city_weather_info = weather_list.get(self.city)
        if city_weather_info is None:
            return False
        
        raw_temp = city_weather_info['temperature']
        raw_weather = city_weather_info['weather']
        current_units = city_weather_info['temperature units']
        
        self.temperature = raw_temp
        self.weather = raw_weather
        
        final_temp = self.converter.convert(raw_temp, current_units, tmp_units)
        return final_temp, raw_weather
    
    def set_city(self, city):
        self.city = city

    def celsius_to_fahrenheit(self):
        return (self.temperature * 9/5) + 32

    def fahrenheit_to_celsius(self):
        return (self.temperature - 32) * 5/9
