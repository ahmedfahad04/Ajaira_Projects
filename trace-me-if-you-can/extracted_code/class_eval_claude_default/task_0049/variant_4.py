class JobMarketplace:
    def __init__(self):
        self.job_listings = []
        self.resumes = []
        self._job_index = {}
        self._resume_index = {}

    def post_job(self, job_title, company, requirements):
        job = {"job_title": job_title, "company": company, "requirements": requirements}
        self.job_listings.append(job)
        self._update_job_index(job)

    def remove_job(self, job):
        if job in self.job_listings:
            self.job_listings.remove(job)
            self._rebuild_job_index()

    def submit_resume(self, name, skills, experience):
        resume = {"name": name, "skills": skills, "experience": experience}
        self.resumes.append(resume)
        self._update_resume_index(resume)

    def withdraw_resume(self, resume):
        if resume in self.resumes:
            self.resumes.remove(resume)
            self._rebuild_resume_index()

    def _update_job_index(self, job):
        title_words = job["job_title"].lower().split()
        for word in title_words + [req.lower() for req in job["requirements"]]:
            if word not in self._job_index:
                self._job_index[word] = []
            if job not in self._job_index[word]:
                self._job_index[word].append(job)

    def _rebuild_job_index(self):
        self._job_index = {}
        for job in self.job_listings:
            self._update_job_index(job)

    def _update_resume_index(self, resume):
        for skill in resume["skills"]:
            skill_lower = skill.lower()
            if skill_lower not in self._resume_index:
                self._resume_index[skill_lower] = []
            if resume not in self._resume_index[skill_lower]:
                self._resume_index[skill_lower].append(resume)

    def _rebuild_resume_index(self):
        self._resume_index = {}
        for resume in self.resumes:
            self._update_resume_index(resume)

    def search_jobs(self, criteria):
        criteria_lower = criteria.lower()
        if criteria_lower in self._job_index:
            return list(self._job_index[criteria_lower])
        
        # Fallback to original search if not in index
        matching_jobs = []
        for job_listing in self.job_listings:
            if (criteria_lower in job_listing["job_title"].lower() or 
                criteria_lower in [r.lower() for r in job_listing["requirements"]]):
                matching_jobs.append(job_listing)
        return matching_jobs

    def get_job_applicants(self, job):
        requirements = job["requirements"]
        if not requirements:
            return list(self.resumes)
        
        candidate_resumes = set(self.resumes)
        for requirement in requirements:
            req_lower = requirement.lower()
            if req_lower in self._resume_index:
                requirement_resumes = set(self._resume_index[req_lower])
                candidate_resumes &= requirement_resumes
            else:
                candidate_resumes = set()
                break
        
        return [resume for resume in candidate_resumes 
                if self.matches_requirements(resume, requirements)]

    @staticmethod
    def matches_requirements(resume, requirements):
        resume_skills = set(resume["skills"])
        required_skills = set(requirements)
        return resume_skills.issubset(required_skills)
