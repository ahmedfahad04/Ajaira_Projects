from datetime import datetime


class TimeSlot:
    def __init__(self, start_time, end_time):
        self.start = datetime.strptime(start_time, '%H:%M')
        self.end = datetime.strptime(end_time, '%H:%M')
    
    def contains(self, time_point):
        return self.start <= time_point <= self.end
    
    def overlaps_with(self, other_slot):
        return (self.start <= other_slot.start <= self.end or 
                self.start <= other_slot.end <= self.end)


class Classroom:
    def __init__(self, id):
        self.id = id
        self.courses = []

    def add_course(self, course):
        if not self._course_exists(course):
            self.courses.append(course)

    def remove_course(self, course):
        if self._course_exists(course):
            self.courses.remove(course)

    def _course_exists(self, course):
        return course in self.courses

    def is_free_at(self, check_time):
        check_dt = datetime.strptime(check_time, '%H:%M')
        
        for course in self.courses:
            slot = TimeSlot(course['start_time'], course['end_time'])
            if slot.contains(check_dt):
                return False
        return True

    def check_course_conflict(self, new_course):
        new_slot = TimeSlot(new_course['start_time'], new_course['end_time'])
        
        for course in self.courses:
            existing_slot = TimeSlot(course['start_time'], course['end_time'])
            if existing_slot.overlaps_with(new_slot):
                return False
        return True
