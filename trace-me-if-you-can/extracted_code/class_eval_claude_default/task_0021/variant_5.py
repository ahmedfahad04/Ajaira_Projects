from datetime import datetime


class Classroom:
    def __init__(self, id):
        self.id = id
        self._course_registry = {}

    @property 
    def courses(self):
        return list(self._course_registry.values())

    def add_course(self, course):
        course_id = id(course)
        if course_id not in self._course_registry:
            self._course_registry[course_id] = course

    def remove_course(self, course):
        course_id = id(course)
        self._course_registry.pop(course_id, None)

    def is_free_at(self, check_time):
        check_time = datetime.strptime(check_time, '%H:%M')

        busy_periods = [
            (datetime.strptime(course['start_time'], '%H:%M'),
             datetime.strptime(course['end_time'], '%H:%M'))
            for course in self.courses
        ]

        return all(not (start <= check_time <= end) for start, end in busy_periods)

    def check_course_conflict(self, new_course):
        new_start_time = datetime.strptime(new_course['start_time'], '%H:%M')
        new_end_time = datetime.strptime(new_course['end_time'], '%H:%M')

        existing_periods = [
            (datetime.strptime(course['start_time'], '%H:%M'),
             datetime.strptime(course['end_time'], '%H:%M'))
            for course in self.courses
        ]

        overlaps = [
            start <= new_start_time <= end or start <= new_end_time <= end
            for start, end in existing_periods
        ]

        return not any(overlaps)
