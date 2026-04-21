from datetime import datetime, timedelta
from collections import defaultdict

class CalendarUtil:
    def __init__(self):
        self._events_by_date = defaultdict(list)
        self._all_events = set()

    def add_event(self, event):
        if id(event) not in self._all_events:
            self._all_events.add(id(event))
            date_key = event['date'].date()
            self._events_by_date[date_key].append(event)

    def remove_event(self, event):
        if id(event) in self._all_events:
            self._all_events.remove(id(event))
            date_key = event['date'].date()
            if event in self._events_by_date[date_key]:
                self._events_by_date[date_key].remove(event)

    @property
    def events(self):
        result = []
        for event_list in self._events_by_date.values():
            result.extend(event_list)
        return result

    def get_events(self, date):
        return list(self._events_by_date[date.date()])

    def is_available(self, start_time, end_time):
        return not any(start_time < event['end_time'] and end_time > event['start_time'] 
                      for event in self.events)

    def get_available_slots(self, date):
        time_slots = [datetime(date.year, date.month, date.day, hour, 0) 
                     for hour in range(24)]
        
        return [(slot, slot + timedelta(hours=1)) 
                for slot in time_slots 
                if self.is_available(slot, slot + timedelta(hours=1))]

    def get_upcoming_events(self, num_events):
        now = datetime.now()
        return [event for event in sorted(self.events, key=lambda e: e['start_time'])
                if event['start_time'] >= now][:num_events]
