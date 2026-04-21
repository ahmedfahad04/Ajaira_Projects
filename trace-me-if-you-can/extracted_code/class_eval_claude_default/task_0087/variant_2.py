import datetime
import time

class TimeUtils:

    def __init__(self):
        self.datetime = datetime.datetime.now()

    def get_current_time(self):
        return f"{self.datetime.hour:02d}:{self.datetime.minute:02d}:{self.datetime.second:02d}"

    def get_current_date(self):
        return f"{self.datetime.year}-{self.datetime.month:02d}-{self.datetime.day:02d}"

    def add_seconds(self, seconds):
        new_datetime = self.datetime + datetime.timedelta(seconds=seconds)
        return f"{new_datetime.hour:02d}:{new_datetime.minute:02d}:{new_datetime.second:02d}"

    def string_to_datetime(self, string):
        return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S")

    def datetime_to_string(self, dt):
        return f"{dt.year}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"

    def get_minutes(self, string_time1, string_time2):
        parsers = [self.string_to_datetime(t) for t in [string_time1, string_time2]]
        return round((parsers[1] - parsers[0]).seconds / 60)

    def get_format_time(self, year, month, day, hour, minute, second):
        dt = datetime.datetime(year, month, day, hour, minute, second)
        return self.datetime_to_string(dt)
