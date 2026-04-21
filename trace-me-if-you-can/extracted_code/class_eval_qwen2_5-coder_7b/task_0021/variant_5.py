from datetime import datetime

class Room:
    def __init__(self, identifier):
        self.identifier = identifier
        self.lessons = []

    def enroll_lesson(self, lesson):
        if lesson not in self.lessons:
            self.lessons.append(lesson)

    def cancel_lesson(self, lesson):
        if lesson in self.lessons:
            self.lessons.remove(lesson)

    def is_available_at(self, check_time):
        check_time = datetime.strptime(check_time, '%H:%M')

        for lesson in self.lessons:
            if datetime.strptime(lesson['start'], '%H:%M') <= check_time <= datetime.strptime(lesson['end'], '%H:%M'):
                return False
        return True

    def verify_lesson_conflict(self, proposed_lesson):
        proposed_start = datetime.strptime(proposed_lesson['start'], '%H:%M')
        proposed_end = datetime.strptime(proposed_lesson['end'], '%H:%M')

        for lesson in self.lessons:
            start = datetime.strptime(lesson['start'], '%H:%M')
            end = datetime.strptime(lesson['end'], '%H:%M')
            if start <= proposed_start and end >= proposed_start:
                return False
            if start <= proposed_end and end >= proposed_end:
                return False
        return True
