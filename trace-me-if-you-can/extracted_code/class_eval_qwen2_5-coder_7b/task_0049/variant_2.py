class EmploymentMarket:
        def __init__(self):
            self.positions = []
            self.applicants = []

        def post_opening(self, opening_title, company_name, job_requirements):
            opening = {"opening_title": opening_title, "company_name": company_name, "job_requirements": job_requirements}
            self.positions.append(opening)

        def delete_opening(self, opening):
            self.positions.remove(opening)

        def submit_profile(self, individual_name, professional_skills, past_work):
            profile = {"individual_name": individual_name, "professional_skills": professional_skills, "past_work": past_work}
            self.applicants.append(profile)

        def remove_profile(self, profile):
            self.applicants.remove(profile)

        def locate_openings(self, search_keyword):
            relevant_openings = []
            for opening in self.positions:
                if search_keyword.lower() in opening["opening_title"].lower() or any(search_keyword.lower() in requirement.lower() for requirement in opening["job_requirements"]):
                    relevant_openings.append(opening)
            return relevant_openings

        def extract_applicants(self, opening):
            potential_applicants = []
            for profile in self.applicants:
                if self.is_eligible(profile, opening["job_requirements"]):
                    potential_applicants.append(profile)
            return potential_applicants

        @staticmethod
        def is_eligible(profile, job_requirements):
            for skill in profile["professional_skills"]:
                if skill not in job_requirements:
                    return False
            return True
