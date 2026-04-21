import datetime
import time

class TimeUtils:

    def __init__(self):
        self.datetime = datetime.datetime.now()

    def get_current_time(self):
        return self._extract_time_string(self.datetime)

    def get_current_date(self):
        return self._extract_date_string(self.datetime)

    def add_seconds(self, seconds):
        modified_datetime = self._create_offset_datetime(seconds)
        return self._extract_time_string(modified_datetime)

    def string_to_datetime(self, string):
        return self._parse_datetime_string(string)

    def datetime_to_string(self, datetime_obj):
        return self._serialize_datetime(datetime_obj)

    def get_minutes(self, string_time1, string_time2):
        dt1 = self._parse_datetime_string(string_time1)
        dt2 = self._parse_datetime_string(string_time2)
        return self._calculate_minute_difference(dt1, dt2)

    def get_format_time(self, year, month, day, hour, minute, second):
        constructed_dt = self._build_datetime(year, month, day, hour, minute, second)
        return self._serialize_datetime(constructed_dt)

    def _extract_time_string(self, dt):
        return dt.strftime("%H:%M:%S")

    def _extract_date_string(self, dt):
        return dt.strftime("%Y-%m-%d")

    def _create_offset_datetime(self, seconds):
        return self.datetime + datetime.timedelta(seconds=seconds)

    def _parse_datetime_string(self, string):
        return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S")

    def _serialize_datetime(self, dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _calculate_minute_difference(self, dt1, dt2):
        return round((dt2 - dt1).seconds / 60)

    def _build_datetime(self, year, month, day, hour, minute, second):
        return datetime.datetime(year, month, day, hour, minute, second)
