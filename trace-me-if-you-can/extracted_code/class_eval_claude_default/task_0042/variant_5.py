class Hotel:
    def __init__(self, name, rooms):
        self.name = name
        self.available_rooms = rooms
        self.booked_rooms = {}

    def book_room(self, room_type, room_number, name):
        def validate_room_type():
            return room_type in self.available_rooms.keys()
        
        def can_book_rooms():
            return room_number <= self.available_rooms[room_type]
        
        def has_partial_availability():
            return self.available_rooms[room_type] != 0
        
        def execute_booking():
            if room_type not in self.booked_rooms.keys():
                self.booked_rooms[room_type] = {}
            self.booked_rooms[room_type][name] = room_number
            self.available_rooms[room_type] -= room_number
            return "Success!"

        if not validate_room_type():
            return False

        if can_book_rooms():
            return execute_booking()
        elif has_partial_availability():
            return self.available_rooms[room_type]
        else:
            return False

    def check_in(self, room_type, room_number, name):
        def has_booking():
            return (room_type in self.booked_rooms.keys() and 
                   name in self.booked_rooms[room_type])
        
        def process_check_in():
            booked_count = self.booked_rooms[room_type][name]
            
            if room_number > booked_count:
                return False
            elif room_number == booked_count:
                self.booked_rooms[room_type].pop(name)
            else:
                self.booked_rooms[room_type][name] -= room_number

        if not has_booking():
            return False
        
        return process_check_in()

    def check_out(self, room_type, room_number):
        if room_type in self.available_rooms:
            self.available_rooms[room_type] += room_number
        else:
            self.available_rooms[room_type] = room_number

    def get_available_rooms(self, room_type):
        return self.available_rooms[room_type]
