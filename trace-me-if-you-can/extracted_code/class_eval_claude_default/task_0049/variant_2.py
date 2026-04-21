class JobMarketplace:
    def __init__(self):
        self._jobs = {}
        self._resumes = {}
        self._job_counter = 0
        self._resume_counter = 0

    def post_job(self, job_title, company, requirements):
        job_id = self._job_counter
        self._jobs[job_id] = {"job_title": job_title, "company": company, "requirements": requirements}
        self._job_counter += 1
        return job_id

    def remove_job(self, job):
        job_id = next((k for k, v in self._jobs.items() if v == job), None)
        if job_id is not None:
            del self._jobs[job_id]

    def submit_resume(self, name, skills, experience):
        resume_id = self._resume_counter
        self._resumes[resume_id] = {"name": name, "skills": skills, "experience": experience}
        self._resume_counter += 1
        return resume_id

    def withdraw_resume(self, resume):
        resume_id = next((k for k, v in self._resumes.items() if v == resume), None)
        if resume_id is not None:
            del self._resumes[resume_id]

    @property
    def job_listings(self):
        return list(self._jobs.values())

    @property
    def resumes(self):
        return list(self._resumes.values())

    def search_jobs(self, criteria):
        criteria_lower = criteria.lower()
        results = []
        for job in self._jobs.values():
            title_match = criteria_lower in job["job_title"].lower()
            req_match = criteria_lower in [r.lower() for r in job["requirements"]]
            if title_match or req_match:
                results.append(job)
        return results

    def get_job_applicants(self, job):
        return [resume for resume in self._resumes.values() 
                if self.matches_requirements(resume, job["requirements"])]

    @staticmethod
    def matches_requirements(resume, requirements):
        skill_set = set(resume["skills"])
        req_set = set(requirements)
        return skill_set.issubset(req_set)
