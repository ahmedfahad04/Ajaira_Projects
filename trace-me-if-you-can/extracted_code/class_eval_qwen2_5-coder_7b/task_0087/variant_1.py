import datetime

   class TimeUtility:
       def __init__(self):
           self.current_time = datetime.datetime.now()

       def get_time(self):
           return self.current_time.strftime("%H:%M:%S")

       def get_date(self):
           return self.current_time.strftime("%Y-%m-%d")

       def add_seconds(self, seconds):
           new_time = self.current_time + datetime.timedelta(seconds=seconds)
           return new_time.strftime("%H:%M:%S")

       def parse_time(self, time_str):
           return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

       def format_time(self, time_obj):
           return time_obj.strftime("%Y-%m-%d %H:%M:%S")

       def minutes_between(self, time_str1, time_str2):
           time1 = self.parse_time(time_str1)
           time2 = self.parse_time(time_str2)
           return round((time2 - time1).seconds / 60)

       def make_time(self, year, month, day, hour, minute, second):
           return datetime.datetime(year, month, day, hour, minute, second).strftime("%Y-%m-%d %H:%M:%S")
