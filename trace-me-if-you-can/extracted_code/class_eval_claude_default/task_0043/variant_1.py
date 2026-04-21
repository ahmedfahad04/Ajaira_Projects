class HRManagementSystem:
    def __init__(self):
        self.employees = {}

    def add_employee(self, employee_id, name, position, department, salary):
        if self._employee_exists(employee_id):
            return False
        
        self.employees[employee_id] = self._create_employee_record(name, position, department, salary)
        return True

    def remove_employee(self, employee_id):
        return self.employees.pop(employee_id, None) is not None

    def update_employee(self, employee_id: int, employee_info: dict):
        employee = self.get_employee(employee_id)
        if not employee:
            return False
        
        if not self._validate_update_keys(employee, employee_info):
            return False
            
        employee.update(employee_info)
        return True

    def get_employee(self, employee_id):
        return self.employees.get(employee_id, False)

    def list_employees(self):
        return {emp_id: self._format_employee_details(emp_id, emp_info) 
                for emp_id, emp_info in self.employees.items()} if self.employees else {}

    def _employee_exists(self, employee_id):
        return employee_id in self.employees

    def _create_employee_record(self, name, position, department, salary):
        return {'name': name, 'position': position, 'department': department, 'salary': salary}

    def _validate_update_keys(self, employee, employee_info):
        return all(key in employee for key in employee_info.keys())

    def _format_employee_details(self, employee_id, employee_info):
        details = {"employee_ID": employee_id}
        details.update(employee_info)
        return details
