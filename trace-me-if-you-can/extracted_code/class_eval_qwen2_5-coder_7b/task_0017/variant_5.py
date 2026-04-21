from datetime import datetime, timedelta

class EventCalendar:
    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)

    def delete_event(self, event):
        if event in self.events:
            self.events.remove(event)

    def get_events_for_date(self, date):
        events_on_date = [event for event in self.events if event['date'].date() == date.date()]
        return events_on_date

    def is_slot_free(self, start_time, end_time):
        return all(not (start_time < event['end_time'] and end_time > event['start_time']) for event in self.events)

    def find_available_slots(self, date):
        available_slots = []
        day_start = datetime(date.year, date.month, date.day, 0, 0)
        day_end = datetime(date.year, date.month, date.day, 23, 59)

        while day_start < day_end:
            slot_end = day_start + timedelta(minutes=60)
            if self.is_slot_free(day_start, slot_end):
                available_slots.append((day_start, slot_end))
            day_start += timedelta(minutes=60)

        return available_slots

    def get_first_n_events(self, n):
        now = datetime.now()
        upcoming_events = [event for event in self.events if event['start_time'] >= now]
        return upcoming_events[:n]
