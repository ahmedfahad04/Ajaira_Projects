from datetime import datetime


class Classroom:
    def __init__(self, id):
        self.id = id
        self.courses = set()

    def add_course(self, course):
        # Convert dict to frozenset for hashability in set
        course_key = frozenset(course.items())
        self.courses.add(course_key)

    def remove_course(self, course):
        course_key = frozenset(course.items())
        self.courses.discard(course_key)

    def is_free_at(self, check_time):
        check_time = datetime.strptime(check_time, '%H:%M')
        
        return not any(
            datetime.strptime(dict(course)['start_time'], '%H:%M') <= check_time <= 
            datetime.strptime(dict(course)['end_time'], '%H:%M')
            for course in self.courses
        )

    def check_course_conflict(self, new_course):
        new_start = datetime.strptime(new_course['start_time'], '%H:%M')
        new_end = datetime.strptime(new_course['end_time'], '%H:%M')

        return not any(
            (datetime.strptime(dict(course)['start_time'], '%H:%M') <= new_start <= 
             datetime.strptime(dict(course)['end_time'], '%H:%M')) or
            (datetime.strptime(dict(course)['start_time'], '%H:%M') <= new_end <= 
             datetime.strptime(dict(course)['end_time'], '%H:%M'))
            for course in self.courses
        )
