class OrganizationalManagement:
    def __init__(self):
        self.personnel = {}

    def add_personnel(self, person_id, name, role, department, salary):
        if person_id in self.personnel:
            return False
        else:
            self.personnel[person_id] = {'name': name, 'role': role, 'department': department, 'salary': salary}
            return True

    def terminate_personnel(self, person_id):
        if person_id in self.personnel:
            del self.personnel[person_id]
            return True
        else:
            return False

    def update_personnel_info(self, person_id, info):
        personnel_member = self.get_personnel(person_id)
        if personnel_member is False:
            return False
        else:
            for field, value in info.items():
                if field not in personnel_member:
                    return False
            for field, value in info.items():
                personnel_member[field] = value
            return True

    def get_personnel(self, person_id):
        return self.personnel.get(person_id, False)

    def list_all_personnel(self):
        personnel_data = {}
        if self.personnel:
            for person_id, personnel_info in self.personnel.items():
                personnel_details = {"personnel_ID": person_id}
                for key, value in personnel_info.items():
                    personnel_details[key] = value
                personnel_data[person_id] = personnel_details
        return personnel_data
