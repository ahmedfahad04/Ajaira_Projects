from datetime import datetime


class EducationalRoom:
    def __init__(self, room_number):
        self.room_number = room_number
        self.lessons = []

    def add_lesson(self, lesson):
        if lesson not in self.lessons:
            self.lessons.append(lesson)

    def remove_lesson(self, lesson):
        if lesson in self.lessons:
            self.lessons.remove(lesson)

    def check_availability_at(self, time_to_check):
        time_to_check = datetime.strptime(time_to_check, '%H:%M')

        for lesson in self.lessons:
            if datetime.strptime(lesson['start_time'], '%H:%M') <= time_to_check <= datetime.strptime(lesson['end_time'],
                                                                                                   '%H:%M'):
                return False
        return True

    def detect_lesson_conflict(self, potential_lesson):
        potential_start_time = datetime.strptime(potential_lesson['start_time'], '%H:%M')
        potential_end_time = datetime.strptime(potential_lesson['end_time'], '%H:%M')

        is_conflicting = False
        for lesson in self.lessons:
            start_time = datetime.strptime(lesson['start_time'], '%H:%M')
            end_time = datetime.strptime(lesson['end_time'], '%H:%M')
            if start_time <= potential_start_time and end_time >= potential_start_time:
                is_conflicting = True
            if start_time <= potential_end_time and end_time >= potential_end_time:
                is_conflicting = True
        return is_conflicting
