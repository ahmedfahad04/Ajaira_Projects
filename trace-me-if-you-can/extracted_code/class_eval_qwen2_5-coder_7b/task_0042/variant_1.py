class HotelManagement:
    def __init__(self, hotel_name, room_counts):
        self.hotel_name = hotel_name
        self.room_inventory = room_counts
        self.reservations = {}

    def allocate_room(self, room_type, room_count, guest_name):
        if room_type not in self.room_inventory or room_count > self.room_inventory[room_type]:
            return False
        if room_type not in self.reservations:
            self.reservations[room_type] = {}
        self.reservations[room_type][guest_name] = room_count
        self.room_inventory[room_type] -= room_count
        return "Room booked successfully!"

    def register_guest(self, room_type, room_count, guest_name):
        if room_type not in self.reservations:
            return False
        if guest_name in self.reservations[room_type]:
            if room_count > self.reservations[room_type][guest_name]:
                return False
            elif room_count == self.reservations[room_type][guest_name]:
                del self.reservations[room_type][guest_name]
            else:
                self.reservations[room_type][guest_name] -= room_count

    def return_room(self, room_type, room_count):
        if room_type in self.room_inventory:
            self.room_inventory[room_type] += room_count
        else:
            self.room_inventory[room_type] = room_count

    def get_free_rooms(self, room_type):
        return self.room_inventory.get(room_type, 0)
