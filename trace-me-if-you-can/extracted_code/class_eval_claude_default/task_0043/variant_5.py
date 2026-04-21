class HRManagementSystem:
    def __init__(self):
        self.employees = {}

    def add_employee(self, employee_id, name, position, department, salary):
        success = employee_id not in self.employees
        
        if success:
            self.employees.setdefault(employee_id, {}).update({
                'name': name,
                'position': position,
                'department': department,
                'salary': salary
            })
        
        return success

    def remove_employee(self, employee_id):
        try:
            self.employees.pop(employee_id)
            return True
        except KeyError:
            return False

    def update_employee(self, employee_id: int, employee_info: dict):
        target_employee = self.get_employee(employee_id)
        
        if target_employee == False:
            return False
        
        # Two-pass approach: validate then update
        validation_passed = True
        for key in employee_info:
            if key not in target_employee:
                validation_passed = False
                break
        
        if validation_passed:
            target_employee.update(employee_info)
        
        return validation_passed

    def get_employee(self, employee_id):
        return self.employees.get(employee_id, False)

    def list_employees(self):
        employee_roster = {}
        
        for emp_id, emp_data in self.employees.items():
            enhanced_record = dict(emp_data)
            enhanced_record["employee_ID"] = emp_id
            employee_roster[emp_id] = enhanced_record
        
        return employee_roster
