from datetime import datetime, timedelta

class EventManager:
    def __init__(self, events=None):
        self._event_storage = events or []
    
    def __iadd__(self, event):
        self._event_storage.append(event)
        return self
    
    def __isub__(self, event):
        if event in self._event_storage:
            self._event_storage.remove(event)
        return self
    
    def __iter__(self):
        return iter(self._event_storage)
    
    def __len__(self):
        return len(self._event_storage)

class CalendarUtil:
    def __init__(self):
        self._event_manager = EventManager()

    @property
    def events(self):
        return list(self._event_manager)

    def add_event(self, event):
        self._event_manager += event

    def remove_event(self, event):
        self._event_manager -= event

    def get_events(self, date):
        target_date = date.date()
        events_on_date = []
        
        for event in self._event_manager:
            if event['date'].date() == target_date:
                events_on_date.append(event)
        
        return events_on_date

    def is_available(self, start_time, end_time):
        for event in self._event_manager:
            event_start, event_end = event['start_time'], event['end_time']
            if not (end_time <= event_start or start_time >= event_end):
                return False
        return True

    def get_available_slots(self, date):
        available_slots = []
        current_hour = datetime(date.year, date.month, date.day, 0, 0)
        
        for _ in range(24):
            slot_end = current_hour + timedelta(hours=1)
            if self.is_available(current_hour, slot_end):
                available_slots.append((current_hour, slot_end))
            current_hour = slot_end
        
        return available_slots

    def get_upcoming_events(self, num_events):
        now = datetime.now()
        upcoming = []
        
        for event in self._event_manager:
            if len(upcoming) == num_events:
                break
            if event['start_time'] >= now:
                upcoming.append(event)
        
        return upcoming
