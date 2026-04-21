from datetime import datetime, timedelta

class Timetable:
    def __init__(self):
        self.entries = []

    def add_entry(self, entry):
        self.entries.append(entry)

    def remove_entry(self, entry):
        if entry in self.entries:
            self.entries.remove(entry)

    def get_entries_for_date(self, date):
        entries_on_date = [entry for entry in self.entries if entry['date'].date() == date.date()]
        return entries_on_date

    def is_time_slot_available(self, start_time, end_time):
        return all(not (start_time < entry['end_time'] and end_time > entry['start_time']) for entry in self.entries)

    def find_available_time_slots(self, date):
        available_slots = []
        day_start = datetime(date.year, date.month, date.day, 0, 0)
        day_end = datetime(date.year, date.month, date.day, 23, 59)

        while day_start < day_end:
            slot_end = day_start + timedelta(minutes=60)
            if self.is_time_slot_available(day_start, slot_end):
                available_slots.append((day_start, slot_end))
            day_start += timedelta(minutes=60)

        return available_slots

    def get_next_events(self, quantity):
        now = datetime.now()
        upcoming_events = [entry for entry in self.entries if entry['start_time'] >= now]
        return upcoming_events[:quantity]
