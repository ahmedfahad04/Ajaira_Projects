import datetime

   class TimeUtility:
       @property
       def current_time(self):
           return datetime.datetime.now()

       def format_time(self, time_obj: datetime, fmt: str = "%H:%M:%S") -> str:
           return time_obj.strftime(fmt)

       def parse_time(self, time_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
           return datetime.datetime.strptime(time_str, fmt)

       @property
       def current_time_str(self):
           return self.format_time(self.current_time)

       @property
       def current_date_str(self):
           return self.format_time(self.current_time, fmt="%Y-%m-%d")

       def add_seconds(self, seconds: int) -> str:
           new_time = self.current_time + datetime.timedelta(seconds=seconds)
           return self.format_time(new_time)

       def minutes_difference(self, time_str1: str) -> int:
           time1 = self.parse_time(time_str1)
           return round((time1 - self.current_time).seconds / 60)

       def make_time(self, year: int, month: int, day: int, hour: int, minute: int, second: int) -> str:
           time_obj = datetime.datetime(year, month, day, hour, minute, second)
           return self.format_time(time_obj)
