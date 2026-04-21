from datetime import datetime


class Classroom:
    def __init__(self, id):
        self.id = id
        self.courses = []

    def add_course(self, course):
        try:
            self.courses.index(course)
        except ValueError:
            self.courses.append(course)

    def remove_course(self, course):
        try:
            self.courses.remove(course)
        except ValueError:
            pass

    def _parse_time(self, time_str):
        return datetime.strptime(time_str, '%H:%M')

    def _time_overlaps(self, start1, end1, start2, end2):
        return start1 <= start2 <= end1 or start1 <= end2 <= end1

    def is_free_at(self, check_time):
        check_dt = self._parse_time(check_time)
        
        for course in self.courses:
            start_dt = self._parse_time(course['start_time'])
            end_dt = self._parse_time(course['end_time'])
            if start_dt <= check_dt <= end_dt:
                return False
        return True

    def check_course_conflict(self, new_course):
        new_start = self._parse_time(new_course['start_time'])
        new_end = self._parse_time(new_course['end_time'])

        for course in self.courses:
            existing_start = self._parse_time(course['start_time'])
            existing_end = self._parse_time(course['end_time'])
            
            if self._time_overlaps(existing_start, existing_end, new_start, new_end):
                return False
        return True
