from datetime import datetime, timedelta
import bisect

class CalendarUtil:
    def __init__(self):
        self.events = []
        self._sorted_events = []

    def add_event(self, event):
        self.events.append(event)
        bisect.insort(self._sorted_events, event, key=lambda x: x['start_time'])

    def remove_event(self, event):
        try:
            self.events.remove(event)
            self._sorted_events.remove(event)
        except ValueError:
            pass

    def get_events(self, date):
        target_date = date.date()
        return list(filter(lambda event: event['date'].date() == target_date, self.events))

    def is_available(self, start_time, end_time):
        def overlaps(event):
            return start_time < event['end_time'] and end_time > event['start_time']
        
        return not any(map(overlaps, self.events))

    def get_available_slots(self, date):
        base_datetime = datetime(date.year, date.month, date.day)
        hour_offsets = range(24)
        
        def create_slot(hour):
            start = base_datetime + timedelta(hours=hour)
            end = start + timedelta(hours=1)
            return (start, end) if self.is_available(start, end) else None
        
        slots = map(create_slot, hour_offsets)
        return [slot for slot in slots if slot is not None]

    def get_upcoming_events(self, num_events):
        now = datetime.now()
        upcoming = []
        
        for event in self._sorted_events:
            if event['start_time'] >= now:
                upcoming.append(event)
                if len(upcoming) >= num_events:
                    break
        
        return upcoming
