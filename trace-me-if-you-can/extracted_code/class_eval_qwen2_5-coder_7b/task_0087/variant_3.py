from datetime import datetime, timedelta
   from typing import Union

   class TimeHandler:
       def __init__(self):
           self.current_time = datetime.now()

       def format(self, dt: datetime, fmt: str = "%H:%M:%S") -> str:
           return dt.strftime(fmt)

       def parse(self, time_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
           return datetime.strptime(time_str, fmt)

       def get_time(self) -> str:
           return self.format(self.current_time)

       def get_date(self) -> str:
           return self.format(self.current_time, fmt="%Y-%m-%d")

       def add_seconds(self, seconds: int) -> str:
           new_time = self.current_time + timedelta(seconds=seconds)
           return self.format(new_time)

       def minutes_difference(self, time_str1: str, time_str2: str) -> int:
           time1 = self.parse(time_str1)
           time2 = self.parse(time_str2)
           return round((time2 - time1).seconds / 60)

       def make_time(self, year: int, month: int, day: int, hour: int, minute: int, second: int) -> str:
           time_obj = datetime(year, month, day, hour, minute, second)
           return self.format(time_obj)
