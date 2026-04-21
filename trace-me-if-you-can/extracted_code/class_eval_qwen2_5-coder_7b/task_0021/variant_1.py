from datetime import datetime


class LearningSpace:
    def __init__(self, identifier):
        self.identifier = identifier
        self.subjects = []

    def include_subject(self, subject):
        if subject not in self.subjects:
            self.subjects.append(subject)

    def exclude_subject(self, subject):
        if subject in self.subjects:
            self.subjects.remove(subject)

    def is_available_at(self, examine_time):
        examine_time = datetime.strptime(examine_time, '%H:%M')

        for subject in self.subjects:
            if datetime.strptime(subject['start_time'], '%H:%M') <= examine_time <= datetime.strptime(subject['end_time'],
                                                                                                   '%H:%M'):
                return False
        return True

    def evaluate_subject_conflict(self, new_subject):
        new_start_time = datetime.strptime(new_subject['start_time'], '%H:%M')
        new_end_time = datetime.strptime(new_subject['end_time'], '%H:%M')

        is_conflicted = False
        for subject in self.subjects:
            start_time = datetime.strptime(subject['start_time'], '%H:%M')
            end_time = datetime.strptime(subject['end_time'], '%H:%M')
            if start_time <= new_start_time and end_time >= new_start_time:
                is_conflicted = True
            if start_time <= new_end_time and end_time >= new_end_time:
                is_conflicted = True
        return is_conflicted
