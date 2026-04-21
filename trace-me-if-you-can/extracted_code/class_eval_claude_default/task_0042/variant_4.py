class RoomInventory:
    def __init__(self, rooms):
        self.available = dict(rooms)
        self.bookings = {}

class Hotel:
    def __init__(self, name, rooms):
        self.name = name
        self.inventory = RoomInventory(rooms)

    def book_room(self, room_type, room_number, name):
        available_rooms = self.inventory.available
        
        if room_type not in available_rooms:
            return False

        current_available = available_rooms[room_type]
        
        if room_number <= current_available:
            bookings_by_type = self.inventory.bookings
            if room_type not in bookings_by_type:
                bookings_by_type[room_type] = {}
            
            bookings_by_type[room_type][name] = room_number
            available_rooms[room_type] = current_available - room_number
            return "Success!"
        elif current_available != 0:
            return current_available
        else:
            return False

    def check_in(self, room_type, room_number, name):
        bookings = self.inventory.bookings
        
        if room_type not in bookings:
            return False
        
        type_bookings = bookings[room_type]
        if name not in type_bookings:
            return False
        
        guest_booking = type_bookings[name]
        
        if room_number > guest_booking:
            return False
        elif room_number == guest_booking:
            type_bookings.pop(name)
        else:
            type_bookings[name] = guest_booking - room_number

    def check_out(self, room_type, room_number):
        available = self.inventory.available
        available[room_type] = available.get(room_type, 0) + room_number

    def get_available_rooms(self, room_type):
        return self.inventory.available[room_type]
