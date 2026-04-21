from datetime import datetime, timedelta

   class TimeManager:
       def __init__(self):
           self.current_datetime = datetime.now()

       def format_time(self, time_obj, fmt="%H:%M:%S"):
           return time_obj.strftime(fmt)

       def parse_time(self, time_str, fmt="%Y-%m-%d %H:%M:%S"):
           return datetime.strptime(time_str, fmt)

       def get_current_time(self):
           return self.format_time(self.current_datetime)

       def get_current_date(self):
           return self.format_time(self.current_datetime, fmt="%Y-%m-%d")

       def add_seconds(self, seconds):
           new_time = self.current_datetime + timedelta(seconds=seconds)
           return self.format_time(new_time)

       def minutes_diff(self, time_str1, time_str2):
           time1 = self.parse_time(time_str1)
           time2 = self.parse_time(time_str2)
           return round((time2 - time1).seconds / 60)

       def create_time(self, year, month, day, hour, minute, second):
           time_obj = datetime(year, month, day, hour, minute, second)
           return self.format_time(time_obj)
