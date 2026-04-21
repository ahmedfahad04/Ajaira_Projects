from datetime import datetime, timedelta

class ScheduleManager:
    def __init__(self):
        self.schedule_items = []

    def append_event(self, event):
        self.schedule_items.append(event)

    def delete_event(self, event):
        if event in self.schedule_items:
            self.schedule_items.remove(event)

    def fetch_events(self, date):
        events_on_date = [event for event in self.schedule_items if event['date'].date() == date.date()]
        return events_on_date

    def check_availability(self, start_time, end_time):
        return all(not (start_time < event['end_time'] and end_time > event['start_time']) for event in self.schedule_items)

    def identify_available_slots(self, date):
        available_slots = []
        day_start = datetime(date.year, date.month, date.day, 0, 0)
        day_end = datetime(date.year, date.month, date.day, 23, 59)

        while day_start < day_end:
            slot_end = day_start + timedelta(minutes=60)
            if self.check_availability(day_start, slot_end):
                available_slots.append((day_start, slot_end))
            day_start += timedelta(minutes=60)

        return available_slots

    def get_earliest_events(self, count):
        now = datetime.now()
        upcoming_events = [event for event in self.schedule_items if event['start_time'] >= now]
        return upcoming_events[:count]
