class HumanResources:
    def __init__(self):
        self.staff = {}

    def add_staff_member(self, member_id, name, role, department, salary):
        if member_id in self.staff:
            return False
        else:
            self.staff[member_id] = {'name': name, 'role': role, 'department': department, 'salary': salary}
            return True

    def remove_staff_member(self, member_id):
        if member_id in self.staff:
            del self.staff[member_id]
            return True
        else:
            return False

    def update_staff_info(self, member_id, info):
        staff_member = self.get_staff_member(member_id)
        if staff_member is False:
            return False
        else:
            for key, value in info.items():
                if key not in staff_member:
                    return False
            for key, value in info.items():
                staff_member[key] = value
            return True

    def get_staff_member(self, member_id):
        return self.staff.get(member_id, False)

    def display_staff(self):
        staff_data = {}
        if self.staff:
            for member_id, member_info in self.staff.items():
                member_details = {"staff_ID": member_id}
                for key, value in member_info.items():
                    member_details[key] = value
                staff_data[member_id] = member_details
        return staff_data
