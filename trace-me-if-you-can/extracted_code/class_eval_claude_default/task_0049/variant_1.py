from collections import defaultdict
from functools import reduce

class JobMarketplace:
    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title, company, requirements):
        job = {"job_title": job_title, "company": company, "requirements": requirements}
        self.job_listings.append(job)

    def remove_job(self, job):
        self.job_listings.remove(job)

    def submit_resume(self, name, skills, experience):
        resume = {"name": name, "skills": skills, "experience": experience}
        self.resumes.append(resume)

    def withdraw_resume(self, resume):
        self.resumes.remove(resume)

    def search_jobs(self, criteria):
        criteria_lower = criteria.lower()
        return list(filter(
            lambda job: criteria_lower in job["job_title"].lower() or 
                       any(criteria_lower in req.lower() for req in job["requirements"]),
            self.job_listings
        ))

    def get_job_applicants(self, job):
        return list(filter(
            lambda resume: self.matches_requirements(resume, job["requirements"]),
            self.resumes
        ))

    @staticmethod
    def matches_requirements(resume, requirements):
        return all(skill in requirements for skill in resume["skills"])
