from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Tuple

@dataclass
class WeatherData:
    temperature: float
    weather: str
    units: str

class WeatherSystem:
    def __init__(self, city: str) -> None:
        self.temperature = None
        self.weather = None
        self.city = city
        self.weather_list = {}
    
    def query(self, weather_list: Dict[str, Dict[str, Any]], tmp_units: str = 'celsius') -> Union[bool, Tuple[float, str]]:
        self.weather_list = weather_list
        
        weather_data = self._extract_weather_data(weather_list)
        if not weather_data:
            return False
        
        self.temperature = weather_data.temperature
        self.weather = weather_data.weather
        
        final_temperature = self._get_temperature_in_units(weather_data, tmp_units)
        return final_temperature, self.weather
    
    def _extract_weather_data(self, weather_list: Dict[str, Dict[str, Any]]) -> Optional[WeatherData]:
        if self.city not in weather_list:
            return None
        
        city_info = weather_list[self.city]
        return WeatherData(
            temperature=city_info['temperature'],
            weather=city_info['weather'],
            units=city_info['temperature units']
        )
    
    def _get_temperature_in_units(self, data: WeatherData, target_units: str) -> float:
        if data.units == target_units:
            return data.temperature
        
        conversion_map = {
            ('celsius', 'fahrenheit'): lambda t: (t * 9/5) + 32,
            ('fahrenheit', 'celsius'): lambda t: (t - 32) * 5/9
        }
        
        converter = conversion_map.get((data.units, target_units))
        return converter(data.temperature) if converter else data.temperature
    
    def set_city(self, city: str):
        self.city = city

    def celsius_to_fahrenheit(self):
        return (self.temperature * 9/5) + 32

    def fahrenheit_to_celsius(self):
        return (self.temperature - 32) * 5/9
