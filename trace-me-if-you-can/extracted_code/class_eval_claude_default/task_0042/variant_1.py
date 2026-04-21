from collections import defaultdict

class Hotel:
    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = rooms.copy()
        self.booked_rooms = defaultdict(dict)

    def book_room(self, room_type, room_number, name):
        if room_type not in self.available_rooms:
            return False
        
        available = self.available_rooms[room_type]
        if room_number <= available:
            self.booked_rooms[room_type][name] = room_number
            self.available_rooms[room_type] -= room_number
            return "Success!"
        elif available != 0:
            return available
        else:
            return False

    def check_in(self, room_type, room_number, name):
        if room_type not in self.booked_rooms or name not in self.booked_rooms[room_type]:
            return False
        
        booked_count = self.booked_rooms[room_type][name]
        if room_number > booked_count:
            return False
        elif room_number == booked_count:
            del self.booked_rooms[room_type][name]
        else:
            self.booked_rooms[room_type][name] -= room_number

    def check_out(self, room_type, room_number):
        self.available_rooms[room_type] = self.available_rooms.get(room_type, 0) + room_number

    def get_available_rooms(self, room_type):
        return self.available_rooms[room_type]
