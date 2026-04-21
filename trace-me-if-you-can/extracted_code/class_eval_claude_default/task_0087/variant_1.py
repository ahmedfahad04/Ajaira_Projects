import datetime
from functools import partial

class TimeUtils:
    TIME_FORMAT = "%H:%M:%S"
    DATE_FORMAT = "%Y-%m-%d"
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        self.datetime = datetime.datetime.now()

    def _format_datetime(self, dt, fmt):
        return dt.strftime(fmt)

    def get_current_time(self):
        return self._format_datetime(self.datetime, self.TIME_FORMAT)

    def get_current_date(self):
        return self._format_datetime(self.datetime, self.DATE_FORMAT)

    def add_seconds(self, seconds):
        new_datetime = self.datetime + datetime.timedelta(seconds=seconds)
        return self._format_datetime(new_datetime, self.TIME_FORMAT)

    def string_to_datetime(self, string):
        return datetime.datetime.strptime(string, self.DATETIME_FORMAT)

    def datetime_to_string(self, datetime):
        return self._format_datetime(datetime, self.DATETIME_FORMAT)

    def get_minutes(self, string_time1, string_time2):
        time1, time2 = map(self.string_to_datetime, [string_time1, string_time2])
        return round((time2 - time1).seconds / 60)

    def get_format_time(self, year, month, day, hour, minute, second):
        time_item = datetime.datetime(year, month, day, hour, minute, second)
        return self._format_datetime(time_item, self.DATETIME_FORMAT)
