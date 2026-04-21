class JobExchange:
        def __init__(self):
            self.positions = []
            self.job_seekers = []

        def post_position(self, title, employer, job_requirements):
            position = {"title": title, "employer": employer, "job_requirements": job_requirements}
            self.positions.append(position)

        def remove_position(self, position):
            self.positions.remove(position)

        def submit_application(self, name, abilities, work_experience):
            application = {"name": name, "abilities": abilities, "work_experience": work_experience}
            self.job_seekers.append(application)

        def withdraw_application(self, application):
            self.job_seekers.remove(application)

        def discover_positions(self, term):
            matching_positions = []
            for position in self.positions:
                if term.lower() in position["title"].lower() or any(term.lower() in requirement.lower() for requirement in position["job_requirements"]):
                    matching_positions.append(position)
            return matching_positions

        def find_applicants(self, position):
            qualified_applicants = []
            for application in self.job_seekers:
                if self.is_eligible(application, position["job_requirements"]):
                    qualified_applicants.append(application)
            return qualified_applicants

        @staticmethod
        def is_eligible(application, job_requirements):
            for ability in application["abilities"]:
                if ability not in job_requirements:
                    return False
            return True
