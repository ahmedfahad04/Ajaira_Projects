class Hotel:
    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = {k: v for k, v in rooms.items()}
        self.booked_rooms = {}

    def book_room(self, room_type, room_number, name):
        try:
            available = self.available_rooms[room_type]
        except KeyError:
            return False

        if room_number <= available:
            guest_bookings = self.booked_rooms.setdefault(room_type, {})
            guest_bookings[name] = room_number
            self.available_rooms[room_type] -= room_number
            return "Success!"
        
        return available if available > 0 else False

    def check_in(self, room_type, room_number, name):
        room_bookings = self.booked_rooms.get(room_type, {})
        guest_rooms = room_bookings.get(name, 0)
        
        if guest_rooms == 0:
            return False
        
        if room_number > guest_rooms:
            return False
        
        if room_number == guest_rooms:
            room_bookings.pop(name)
        else:
            room_bookings[name] = guest_rooms - room_number

    def check_out(self, room_type, room_number):
        if room_type in self.available_rooms:
            self.available_rooms[room_type] += room_number
        else:
            self.available_rooms[room_type] = room_number

    def get_available_rooms(self, room_type):
        return self.available_rooms[room_type]
