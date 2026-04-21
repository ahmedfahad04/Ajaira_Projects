import datetime
import time

class TimeUtils:

    def __init__(self):
        self.datetime = datetime.datetime.now()

    def get_current_time(self):
        components = [self.datetime.hour, self.datetime.minute, self.datetime.second]
        return ":".join(f"{comp:02d}" for comp in components)

    def get_current_date(self):
        date_parts = [self.datetime.year, self.datetime.month, self.datetime.day]
        return f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"

    def add_seconds(self, seconds):
        future_time = self.datetime + datetime.timedelta(seconds=seconds)
        time_components = [future_time.hour, future_time.minute, future_time.second]
        return ":".join(f"{comp:02d}" for comp in time_components)

    def string_to_datetime(self, string):
        date_part, time_part = string.split(" ")
        year, month, day = map(int, date_part.split("-"))
        hour, minute, second = map(int, time_part.split(":"))
        return datetime.datetime(year, month, day, hour, minute, second)

    def datetime_to_string(self, dt):
        date_str = f"{dt.year}-{dt.month:02d}-{dt.day:02d}"
        time_str = f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
        return f"{date_str} {time_str}"

    def get_minutes(self, string_time1, string_time2):
        times = [self.string_to_datetime(t) for t in [string_time1, string_time2]]
        time_diff = times[1] - times[0]
        return round(time_diff.seconds / 60)

    def get_format_time(self, year, month, day, hour, minute, second):
        dt_instance = datetime.datetime(year, month, day, hour, minute, second)
        return self.datetime_to_string(dt_instance)
