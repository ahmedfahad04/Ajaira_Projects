from datetime import datetime, timedelta

class CalendarHandler:
    def __init__(self):
        self.events = []

    def add_event_record(self, event):
        self.events.append(event)

    def remove_event_record(self, event):
        if event in self.events:
            self.events.remove(event)

    def get_events_on_date(self, date):
        events_on_date = [event for event in self.events if event['date'].date() == date.date()]
        return events_on_date

    def check_slot_availability(self, start_time, end_time):
        return all(not (start_time < event['end_time'] and end_time > event['start_time']) for event in self.events)

    def identify_available_time_slots(self, date):
        available_slots = []
        day_start = datetime(date.year, date.month, date.day, 0, 0)
        day_end = datetime(date.year, date.month, date.day, 23, 59)

        while day_start < day_end:
            slot_end = day_start + timedelta(minutes=60)
            if self.check_slot_availability(day_start, slot_end):
                available_slots.append((day_start, slot_end))
            day_start += timedelta(minutes=60)

        return available_slots

    def list_upcoming_events(self, count):
        now = datetime.now()
        upcoming_events = [event for event in self.events if event['start_time'] >= now]
        return upcoming_events[:count]
