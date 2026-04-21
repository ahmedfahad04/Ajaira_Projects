from datetime import datetime


class LectureRoom:
    def __init__(self, room_id):
        self.room_id = room_id
        self.sessions = []

    def enroll_session(self, session):
        if session not in self.sessions:
            self.sessions.append(session)

    def withdraw_session(self, session):
        if session in self.sessions:
            self.sessions.remove(session)

    def validate_schedule_at(self, check_time):
        check_time = datetime.strptime(check_time, '%H:%M')

        for session in self.sessions:
            if datetime.strptime(session['start_time'], '%H:%M') <= check_time <= datetime.strptime(session['end_time'],
                                                                                                   '%H:%M'):
                return False
        return True

    def identify_session_conflict(self, proposed_session):
        proposed_start_time = datetime.strptime(proposed_session['start_time'], '%H:%M')
        proposed_end_time = datetime.strptime(proposed_session['end_time'], '%H:%M')

        is_conflicting = False
        for session in self.sessions:
            start_time = datetime.strptime(session['start_time'], '%H:%M')
            end_time = datetime.strptime(session['end_time'], '%H:%M')
            if start_time <= proposed_start_time and end_time >= proposed_start_time:
                is_conflicting = True
            if start_time <= proposed_end_time and end_time >= proposed_end_time:
                is_conflicting = True
        return is_conflicting
