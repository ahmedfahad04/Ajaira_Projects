from datetime import datetime, timedelta
from itertools import takewhile, islice

class CalendarUtil:
    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)

    def remove_event(self, event):
        while event in self.events:
            self.events.remove(event)

    def get_events(self, date):
        return [event for event in self.events 
                if event['date'].date() == date.date()]

    def _has_time_overlap(self, start1, end1, start2, end2):
        return max(start1, start2) < min(end1, end2)

    def is_available(self, start_time, end_time):
        overlapping_events = (
            event for event in self.events 
            if self._has_time_overlap(start_time, end_time, 
                                    event['start_time'], event['end_time'])
        )
        
        try:
            next(overlapping_events)
            return False
        except StopIteration:
            return True

    def get_available_slots(self, date):
        def time_slot_generator(start_date):
            current = datetime(start_date.year, start_date.month, start_date.day, 0, 0)
            end_of_day = current + timedelta(days=1)
            
            while current < end_of_day - timedelta(hours=1):
                yield current, current + timedelta(hours=1)
                current += timedelta(hours=1)
        
        all_slots = time_slot_generator(date)
        available_slots = (
            slot for slot in all_slots 
            if self.is_available(slot[0], slot[1])
        )
        
        return list(available_slots)

    def get_upcoming_events(self, num_events):
        now = datetime.now()
        future_events = (
            event for event in self.events 
            if event['start_time'] >= now
        )
        
        return list(islice(future_events, num_events))
