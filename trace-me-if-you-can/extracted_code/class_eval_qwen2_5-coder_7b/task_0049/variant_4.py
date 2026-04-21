class RecruitmentSystem:
        def __init__(self):
            self.openings = []
            self.candidate_profiles = []

        def advertise_job(self, job_title, employer, required_skills):
            opening = {"job_title": job_title, "employer": employer, "required_skills": required_skills}
            self.openings.append(opening)

        def remove_advertisement(self, opening):
            self.openings.remove(opening)

        def send_resume(self, candidate_name, skills, past_jobs):
            profile = {"candidate_name": candidate_name, "skills": skills, "past_jobs": past_jobs}
            self.candidate_profiles.append(profile)

        def retract_resume(self, profile):
            self.candidate_profiles.remove(profile)

        def find_openings(self, search_term):
            matching_openings = []
            for opening in self.openings:
                if search_term.lower() in opening["job_title"].lower() or any(search_term.lower() in skill.lower() for skill in opening["required_skills"]):
                    matching_openings.append(opening)
            return matching_openings

        def identify_eligible_applicants(self, opening):
            qualified_applicants = []
            for profile in self.candidate_profiles:
                if self.is_eligible(profile, opening["required_skills"]):
                    qualified_applicants.append(profile)
            return qualified_applicants

        @staticmethod
        def is_eligible(profile, required_skills):
            for skill in profile["skills"]:
                if skill not in required_skills:
                    return False
            return True
