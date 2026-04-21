from datetime import datetime, timedelta
from functools import reduce

class CalendarUtil:
    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events = [*self.events, event]

    def remove_event(self, event):
        self.events = [e for e in self.events if e is not event]

    def get_events(self, date):
        date_matcher = lambda event: event['date'].date() == date.date()
        return list(filter(date_matcher, self.events))

    def is_available(self, start_time, end_time):
        conflict_checker = lambda acc, event: acc or (
            start_time < event['end_time'] and end_time > event['start_time']
        )
        has_conflict = reduce(conflict_checker, self.events, False)
        return not has_conflict

    def get_available_slots(self, date):
        def generate_slots(current_time, end_boundary, slots):
            if current_time >= end_boundary:
                return slots
            
            next_hour = current_time + timedelta(hours=1)
            new_slots = slots + [(current_time, next_hour)] if self.is_available(current_time, next_hour) else slots
            
            return generate_slots(current_time + timedelta(hours=1), end_boundary, new_slots)
        
        start_of_day = datetime(date.year, date.month, date.day, 0, 0)
        end_of_day = datetime(date.year, date.month, date.day, 23, 0)
        
        return generate_slots(start_of_day, end_of_day, [])

    def get_upcoming_events(self, num_events):
        current_time = datetime.now()
        future_events = filter(lambda e: e['start_time'] >= current_time, self.events)
        chronological_events = sorted(future_events, key=lambda e: e['start_time'])
        
        return chronological_events[:num_events]
