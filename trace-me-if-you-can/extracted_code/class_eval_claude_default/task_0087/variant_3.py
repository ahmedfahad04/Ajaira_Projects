import datetime
import time
from typing import Union

class TimeUtils:
    
    def __init__(self):
        self.datetime = datetime.datetime.now()
    
    @staticmethod
    def _apply_format(dt_obj: datetime.datetime, pattern: str) -> str:
        return dt_obj.strftime(pattern)
    
    def get_current_time(self) -> str:
        return TimeUtils._apply_format(self.datetime, "%H:%M:%S")
    
    def get_current_date(self) -> str:
        return TimeUtils._apply_format(self.datetime, "%Y-%m-%d")
    
    def add_seconds(self, seconds: int) -> str:
        adjusted_time = self.datetime + datetime.timedelta(seconds=seconds)
        return TimeUtils._apply_format(adjusted_time, "%H:%M:%S")
    
    @staticmethod
    def string_to_datetime(string: str) -> datetime.datetime:
        return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    
    @staticmethod 
    def datetime_to_string(dt: datetime.datetime) -> str:
        return TimeUtils._apply_format(dt, "%Y-%m-%d %H:%M:%S")
    
    def get_minutes(self, string_time1: str, string_time2: str) -> int:
        dt1, dt2 = (TimeUtils.string_to_datetime(t) for t in (string_time1, string_time2))
        return round((dt2 - dt1).seconds / 60)
    
    @staticmethod
    def get_format_time(year: int, month: int, day: int, hour: int, minute: int, second: int) -> str:
        constructed_dt = datetime.datetime(year, month, day, hour, minute, second)
        return TimeUtils.datetime_to_string(constructed_dt)
