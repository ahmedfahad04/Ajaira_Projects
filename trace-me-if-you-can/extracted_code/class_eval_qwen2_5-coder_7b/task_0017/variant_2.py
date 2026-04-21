from datetime import datetime, timedelta

class EventOrganizer:
    def __init__(self):
        self.events = []

    def include_event(self, event):
        self.events.append(event)

    def exclude_event(self, event):
        if event in self.events:
            self.events.remove(event)

    def retrieve_events(self, date):
        events_on_date = [event for event in self.events if event['date'].date() == date.date()]
        return events_on_date

    def is_schedule_free(self, start_time, end_time):
        return not any(start_time < event['end_time'] and end_time > event['start_time'] for event in self.events)

    def find_free_slots(self, date):
        free_slots = []
        day_start = datetime(date.year, date.month, date.day, 0, 0)
        day_end = datetime(date.year, date.month, date.day, 23, 59)

        while day_start < day_end:
            slot_end = day_start + timedelta(minutes=60)
            if self.is_schedule_free(day_start, slot_end):
                free_slots.append((day_start, slot_end))
            day_start += timedelta(minutes=60)

        return free_slots

    def list_future_events(self, count):
        now = datetime.now()
        upcoming_events = [event for event in self.events if event['start_time'] >= now]
        return upcoming_events[:count]
