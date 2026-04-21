class HRManagementSystem:
    def __init__(self):
        self.employees = {}

    def add_employee(self, employee_id, name, position, department, salary):
        employee_exists = employee_id in self.employees
        if not employee_exists:
            self.employees[employee_id] = {
                'name': name,
                'position': position,
                'department': department,
                'salary': salary
            }
        return not employee_exists

    def remove_employee(self, employee_id):
        employee_found = employee_id in self.employees
        if employee_found:
            self.employees.pop(employee_id)
        return employee_found

    def update_employee(self, employee_id: int, employee_info: dict):
        employee = self.get_employee(employee_id)
        if employee is False:
            return False
        
        # Check all keys exist before any updates
        valid_keys = all(key in employee for key in employee_info)
        if not valid_keys:
            return False
        
        # Apply all updates at once
        for key, value in employee_info.items():
            employee[key] = value
        return True

    def get_employee(self, employee_id):
        return self.employees.get(employee_id, False)

    def list_employees(self):
        if not self.employees:
            return {}
        
        result = {}
        for employee_id in self.employees:
            employee_record = {"employee_ID": employee_id}
            employee_record.update(self.employees[employee_id])
            result[employee_id] = employee_record
        return result
