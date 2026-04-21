class HRManagementSystem:
    EMPLOYEE_FIELDS = ['name', 'position', 'department', 'salary']
    
    def __init__(self):
        self.employees = {}

    def add_employee(self, employee_id, name, position, department, salary):
        if self._has_employee(employee_id):
            return False
        
        employee_data = dict(zip(self.EMPLOYEE_FIELDS, [name, position, department, salary]))
        self.employees[employee_id] = employee_data
        return True

    def remove_employee(self, employee_id):
        if not self._has_employee(employee_id):
            return False
        
        del self.employees[employee_id]
        return True

    def update_employee(self, employee_id: int, employee_info: dict):
        if not self._has_employee(employee_id):
            return False
        
        current_employee = self.employees[employee_id]
        
        # Validate all keys first
        for key in employee_info:
            if key not in current_employee:
                return False
        
        # Then update all values
        for key, value in employee_info.items():
            current_employee[key] = value
        
        return True

    def get_employee(self, employee_id):
        return self.employees[employee_id] if self._has_employee(employee_id) else False

    def list_employees(self):
        employee_list = {}
        for emp_id, emp_info in self.employees.items():
            formatted_employee = {"employee_ID": emp_id, **emp_info}
            employee_list[emp_id] = formatted_employee
        return employee_list

    def _has_employee(self, employee_id):
        return employee_id in self.employees
