class Hotel:
    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = dict(rooms)
        self.booked_rooms = {}

    def _ensure_room_type_exists(self, room_type):
        if room_type not in self.booked_rooms:
            self.booked_rooms[room_type] = {}

    def _has_available_rooms(self, room_type):
        return room_type in self.available_rooms

    def book_room(self, room_type, room_number, name):
        if not self._has_available_rooms(room_type):
            return False

        available_count = self.available_rooms[room_type]
        
        if room_number > available_count:
            return False if available_count == 0 else available_count
        
        self._ensure_room_type_exists(room_type)
        self.booked_rooms[room_type][name] = room_number
        self.available_rooms[room_type] -= room_number
        return "Success!"

    def check_in(self, room_type, room_number, name):
        if not (room_type in self.booked_rooms and name in self.booked_rooms[room_type]):
            return False
        
        booked_rooms = self.booked_rooms[room_type][name]
        
        if room_number > booked_rooms:
            return False
        
        if room_number == booked_rooms:
            self.booked_rooms[room_type].pop(name)
        else:
            self.booked_rooms[room_type][name] = booked_rooms - room_number

    def check_out(self, room_type, room_number):
        current_available = self.available_rooms.get(room_type, 0)
        self.available_rooms[room_type] = current_available + room_number

    def get_available_rooms(self, room_type):
        return self.available_rooms[room_type]
