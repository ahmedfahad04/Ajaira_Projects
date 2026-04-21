import datetime
   from typing import Callable

   class TimeUtility:
       def __init__(self):
           self.current_time = datetime.datetime.now()

       def format_time(self, time_obj: datetime, fmt: str = "%H:%M:%S") -> str:
           return time_obj.strftime(fmt)

       def parse_time(self, time_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
           return datetime.datetime.strptime(time_str, fmt)

       get_current_time = lambda self: self.format_time(self.current_time)
       get_current_date = lambda self: self.format_time(self.current_time, fmt="%Y-%m-%d")

       def add_seconds(self, seconds: int) -> str:
           new_time = self.current_time + datetime.timedelta(seconds=seconds)
           return self.format_time(new_time)

       minutes_difference = lambda self, time_str1: round((self.parse_time(time_str1) - self.current_time).seconds / 60)
       make_time = lambda self, year, month, day, hour, minute, second: self.format_time(datetime.datetime(year, month, day, hour, minute, second))

       def format_time_with_year_month_day(self, year: int, month: int, day: int, hour: int, minute: int, second: int) -> str:
           return self.format_time(datetime.datetime(year, month, day, hour, minute, second))
