class HotelBookingSystem:
    def __init__(self, hotel_name, room_availability):
        self.hotel_name = hotel_name
        self.room_availability = room_availability
        self.booked_rooms = {}

    def assign_room(self, room_type, room_count, guest):
        if room_type not in self.room_availability or room_count > self.room_availability[room_type]:
            return False
        if room_type not in self.booked_rooms:
            self.booked_rooms[room_type] = {}
        self.booked_rooms[room_type][guest] = room_count
        self.room_availability[room_type] -= room_count
        return "Room allocated."

    def guest_arrival(self, room_type, room_count, guest):
        if room_type not in self.booked_rooms:
            return False
        if guest in self.booked_rooms[room_type]:
            if room_count > self.booked_rooms[room_type][guest]:
                return False
            elif room_count == self.booked_rooms[room_type][guest]:
                del self.booked_rooms[room_type][guest]
            else:
                self.booked_rooms[room_type][guest] -= room_count

    def free_room(self, room_type, room_count):
        if room_type in self.room_availability:
            self.room_availability[room_type] += room_count
        else:
            self.room_availability[room_type] = room_count

    def check_room_availability(self, room_type):
        return self.room_availability.get(room_type, 0)
