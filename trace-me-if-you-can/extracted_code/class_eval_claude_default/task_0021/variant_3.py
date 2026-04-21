from datetime import datetime
from functools import reduce


class Classroom:
    def __init__(self, id):
        self.id = id
        self.courses = []

    def add_course(self, course):
        if course not in self.courses:
            self.courses = [*self.courses, course]

    def remove_course(self, course):
        self.courses = [c for c in self.courses if c != course]

    def is_free_at(self, check_time):
        check_time = datetime.strptime(check_time, '%H:%M')
        
        conflicts = map(
            lambda course: datetime.strptime(course['start_time'], '%H:%M') <= check_time <= 
                          datetime.strptime(course['end_time'], '%H:%M'),
            self.courses
        )
        
        return not reduce(lambda acc, conflict: acc or conflict, conflicts, False)

    def check_course_conflict(self, new_course):
        new_start_time = datetime.strptime(new_course['start_time'], '%H:%M')
        new_end_time = datetime.strptime(new_course['end_time'], '%H:%M')

        def has_conflict(course):
            start_time = datetime.strptime(course['start_time'], '%H:%M')
            end_time = datetime.strptime(course['end_time'], '%H:%M')
            return ((start_time <= new_start_time <= end_time) or 
                   (start_time <= new_end_time <= end_time))

        conflicts = map(has_conflict, self.courses)
        return not reduce(lambda acc, conflict: acc or conflict, conflicts, False)
