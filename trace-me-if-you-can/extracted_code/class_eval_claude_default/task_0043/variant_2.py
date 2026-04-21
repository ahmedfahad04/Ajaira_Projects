from collections import defaultdict

class HRManagementSystem:
    def __init__(self):
        self.employees = defaultdict(dict)

    def add_employee(self, employee_id, name, position, department, salary):
        try:
            if self.employees[employee_id]:
                return False
        except KeyError:
            pass
        
        self.employees[employee_id] = {
            'name': name,
            'position': position, 
            'department': department,
            'salary': salary
        }
        return True

    def remove_employee(self, employee_id):
        if employee_id not in self.employees:
            return False
        del self.employees[employee_id]
        return True

    def update_employee(self, employee_id: int, employee_info: dict):
        if employee_id not in self.employees:
            return False
        
        current_employee = self.employees[employee_id]
        invalid_keys = [key for key in employee_info if key not in current_employee]
        
        if invalid_keys:
            return False
            
        current_employee.update(employee_info)
        return True

    def get_employee(self, employee_id):
        return self.employees.get(employee_id) or False

    def list_employees(self):
        result = {}
        for emp_id in self.employees:
            emp_data = dict(self.employees[emp_id])
            emp_data["employee_ID"] = emp_id
            result[emp_id] = emp_data
        return result
