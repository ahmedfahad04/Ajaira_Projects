class CompanyHRSystem:
    def __init__(self):
        self.employees = {}

    def add_worker(self, worker_id, name, position, dept, salary):
        if worker_id in self.employees:
            return False
        else:
            self.employees[worker_id] = {'name': name, 'position': position, 'department': dept, 'salary': salary}
            return True

    def dismiss_worker(self, worker_id):
        if worker_id in self.employees:
            del self.employees[worker_id]
            return True
        else:
            return False

    def update_worker_info(self, worker_id, info):
        worker = self.retrieve_worker(worker_id)
        if worker is False:
            return False
        else:
            for key, value in info.items():
                if key not in worker:
                    return False
            for key, value in info.items():
                worker[key] = value
            return True

    def retrieve_worker(self, worker_id):
        return self.employees.get(worker_id, False)

    def display_all_employees(self):
        employee_data = {}
        if self.employees:
            for worker_id, worker_info in self.employees.items():
                worker_details = {"employee_ID": worker_id}
                for key, value in worker_info.items():
                    worker_details[key] = value
                employee_data[worker_id] = worker_details
        return employee_data
