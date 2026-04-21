class HotelSystem:
    def __init__(self, hotel_name, available_rooms):
        self.name = hotel_name
        self.rooms = available_rooms
        self.occupied_rooms = {}

    def reserve_room(self, room_type, room_number, guest):
        if room_type not in self.rooms:
            return False
        if room_number <= self.rooms[room_type]:
            if room_type not in self.occupied_rooms:
                self.occupied_rooms[room_type] = {}
            self.occupied_rooms[room_type][guest] = room_number
            self.rooms[room_type] -= room_number
            return "Room booked."
        return self.rooms[room_type]

    def admit_guest(self, room_type, room_number, guest):
        if room_type not in self.occupied_rooms:
            return False
        if guest in self.occupied_rooms[room_type]:
            if room_number > self.occupied_rooms[room_type][guest]:
                return False
            elif room_number == self.occupied_rooms[room_type][guest]:
                del self.occupied_rooms[room_type][guest]
            else:
                self.occupied_rooms[room_type][guest] -= room_number

    def release_room(self, room_type, room_number):
        if room_type in self.rooms:
            self.rooms[room_type] += room_number
        else:
            self.rooms[room_type] = room_number

    def get_room_count(self, room_type):
        return self.rooms.get(room_type, 0)
