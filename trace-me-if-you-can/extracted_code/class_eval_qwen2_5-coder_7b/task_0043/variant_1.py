class EmployeeManagementSystem:
    def __init__(self):
        self.staff = {}

    def include_employee(self, emp_id, name, role, department, pay):
        if emp_id in self.staff:
            return False
        else:
            self.staff[emp_id] = {'name': name, 'role': role, 'department': department, 'pay': pay}
            return True

    def terminate_employee(self, emp_id):
        if emp_id in self.staff:
            del self.staff[emp_id]
            return True
        else:
            return False

    def modify_employee(self, emp_id, details):
        staff_member = self.fetch_employee(emp_id)
        if staff_member is False:
            return False
        else:
            for field, value in details.items():
                if field not in staff_member:
                    return False
            for field, value in details.items():
                staff_member[field] = value
            return True

    def fetch_employee(self, emp_id):
        return self.staff.get(emp_id, False)

    def view_employees(self):
        staff_data = {}
        if self.staff:
            for emp_id, details in self.staff.items():
                staff_details = {"employee_ID": emp_id}
                for key, value in details.items():
                    staff_details[key] = value
                staff_data[emp_id] = staff_details
        return staff_data
