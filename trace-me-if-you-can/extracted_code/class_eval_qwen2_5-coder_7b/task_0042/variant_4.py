class HotelReservations:
    def __init__(self, hotel_name, room_numbers):
        self.hotel_name = hotel_name
        self.room_numbers = room_numbers
        self.booked = {}

    def reserve(self, room_type, room_count, guest_name):
        if room_type not in self.room_numbers or room_count > self.room_numbers[room_type]:
            return False
        if room_type not in self.booked:
            self.booked[room_type] = {}
        self.booked[room_type][guest_name] = room_count
        self.room_numbers[room_type] -= room_count
        return "Room booked."

    def check_in_guest(self, room_type, room_count, guest_name):
        if room_type not in self.booked:
            return False
        if guest_name in self.booked[room_type]:
            if room_count > self.booked[room_type][guest_name]:
                return False
            elif room_count == self.booked[room_type][guest_name]:
                del self.booked[room_type][guest_name]
            else:
                self.booked[room_type][guest_name] -= room_count

    def check_out_room(self, room_type, room_count):
        if room_type in self.room_numbers:
            self.room_numbers[room_type] += room_count
        else:
            self.room_numbers[room_type] = room_count

    def get_available_room_count(self, room_type):
        return self.room_numbers.get(room_type, 0)
