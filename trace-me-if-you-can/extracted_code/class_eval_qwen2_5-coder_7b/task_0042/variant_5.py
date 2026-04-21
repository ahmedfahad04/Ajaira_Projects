class HotelInventory:
    def __init__(self, hotel_name, room_inventory):
        self.hotel_name = hotel_name
        self.room_inventory = room_inventory
        self.bookings = {}

    def allocate(self, room_type, room_count, guest):
        if room_type not in self.room_inventory or room_count > self.room_inventory[room_type]:
            return False
        if room_type not in self.bookings:
            self.bookings[room_type] = {}
        self.bookings[room_type][guest] = room_count
        self.room_inventory[room_type] -= room_count
        return "Room allocated."

    def admit(self, room_type, room_count, guest):
        if room_type not in self.bookings:
            return False
        if guest in self.bookings[room_type]:
            if room_count > self.bookings[room_type][guest]:
                return False
            elif room_count == self.bookings[room_type][guest]:
                del self.bookings[room_type][guest]
            else:
                self.bookings[room_type][guest] -= room_count

    def return_room_to_inventory(self, room_type, room_count):
        if room_type in self.room_inventory:
            self.room_inventory[room_type] += room_count
        else:
            self.room_inventory[room_type] = room_count

    def get_room_availability(self, room_type):
        return self.room_inventory.get(room_type, 0)
