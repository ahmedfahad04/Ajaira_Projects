class PersonnelSystem:
    def __init__(self):
        self.employees = {}

    def add_person(self, person_id, name, position, department, salary):
        if person_id in self.employees:
            return False
        else:
            self.employees[person_id] = {'name': name, 'position': position, 'department': department, 'salary': salary}
            return True

    def remove_person(self, person_id):
        if person_id in self.employees:
            del self.employees[person_id]
            return True
        else:
            return False

    def update_person_info(self, person_id, info):
        person = self.fetch_person(person_id)
        if person is False:
            return False
        else:
            for key, value in info.items():
                if key not in person:
                    return False
            for key, value in info.items():
                person[key] = value
            return True

    def fetch_person(self, person_id):
        return self.employees.get(person_id, False)

    def list_all_people(self):
        people_data = {}
        if self.employees:
            for person_id, person_info in self.employees.items():
                person_details = {"person_ID": person_id}
                for key, value in person_info.items():
                    person_details[key] = value
                people_data[person_id] = person_details
        return people_data
