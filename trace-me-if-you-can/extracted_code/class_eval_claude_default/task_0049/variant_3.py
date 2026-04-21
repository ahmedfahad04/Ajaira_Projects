class Job:
    def __init__(self, job_title, company, requirements):
        self.job_title = job_title
        self.company = company
        self.requirements = requirements

class Resume:
    def __init__(self, name, skills, experience):
        self.name = name
        self.skills = skills
        self.experience = experience

class JobMarketplace:
    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title, company, requirements):
        job = Job(job_title, company, requirements)
        self.job_listings.append(job)

    def remove_job(self, job):
        # Convert object back to dict format for compatibility
        job_dict = {"job_title": job.job_title, "company": job.company, "requirements": job.requirements}
        matching_job = next((j for j in self.job_listings if self._job_to_dict(j) == job_dict), None)
        if matching_job:
            self.job_listings.remove(matching_job)

    def submit_resume(self, name, skills, experience):
        resume = Resume(name, skills, experience)
        self.resumes.append(resume)

    def withdraw_resume(self, resume):
        # Convert object back to dict format for compatibility
        resume_dict = {"name": resume.name, "skills": resume.skills, "experience": resume.experience}
        matching_resume = next((r for r in self.resumes if self._resume_to_dict(r) == resume_dict), None)
        if matching_resume:
            self.resumes.remove(matching_resume)

    def _job_to_dict(self, job):
        return {"job_title": job.job_title, "company": job.company, "requirements": job.requirements}

    def _resume_to_dict(self, resume):
        return {"name": resume.name, "skills": resume.skills, "experience": resume.experience}

    def search_jobs(self, criteria):
        matching_jobs = []
        criteria_lower = criteria.lower()
        for job in self.job_listings:
            if (criteria_lower in job.job_title.lower() or 
                criteria_lower in [r.lower() for r in job.requirements]):
                matching_jobs.append(self._job_to_dict(job))
        return matching_jobs

    def get_job_applicants(self, job):
        applicants = []
        for resume in self.resumes:
            if self.matches_requirements(self._resume_to_dict(resume), job["requirements"]):
                applicants.append(self._resume_to_dict(resume))
        return applicants

    @staticmethod
    def matches_requirements(resume, requirements):
        return all(skill in requirements for skill in resume["skills"])
