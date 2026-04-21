from typing import List, Dict, Any, Callable

class JobMarketplace:
    def __init__(self):
        self.job_listings = []
        self.resumes = []

    def post_job(self, job_title: str, company: str, requirements: List[str]) -> None:
        job = self._create_job_entry(job_title, company, requirements)
        self._add_to_collection(self.job_listings, job)

    def remove_job(self, job: Dict[str, Any]) -> None:
        self._remove_from_collection(self.job_listings, job)

    def submit_resume(self, name: str, skills: List[str], experience: str) -> None:
        resume = self._create_resume_entry(name, skills, experience)
        self._add_to_collection(self.resumes, resume)

    def withdraw_resume(self, resume: Dict[str, Any]) -> None:
        self._remove_from_collection(self.resumes, resume)

    def search_jobs(self, criteria: str) -> List[Dict[str, Any]]:
        search_predicate = self._create_job_search_predicate(criteria)
        return self._filter_collection(self.job_listings, search_predicate)

    def get_job_applicants(self, job: Dict[str, Any]) -> List[Dict[str, Any]]:
        applicant_predicate = lambda resume: self.matches_requirements(resume, job["requirements"])
        return self._filter_collection(self.resumes, applicant_predicate)

    @staticmethod
    def _create_job_entry(job_title: str, company: str, requirements: List[str]) -> Dict[str, Any]:
        return {"job_title": job_title, "company": company, "requirements": requirements}

    @staticmethod
    def _create_resume_entry(name: str, skills: List[str], experience: str) -> Dict[str, Any]:
        return {"name": name, "skills": skills, "experience": experience}

    @staticmethod
    def _add_to_collection(collection: List[Any], item: Any) -> None:
        collection.append(item)

    @staticmethod
    def _remove_from_collection(collection: List[Any], item: Any) -> None:
        collection.remove(item)

    @staticmethod
    def _filter_collection(collection: List[Any], predicate: Callable[[Any], bool]) -> List[Any]:
        return [item for item in collection if predicate(item)]

    @staticmethod
    def _create_job_search_predicate(criteria: str) -> Callable[[Dict[str, Any]], bool]:
        criteria_lower = criteria.lower()
        return lambda job: (criteria_lower in job["job_title"].lower() or 
                           criteria_lower in [r.lower() for r in job["requirements"]])

    @staticmethod
    def matches_requirements(resume: Dict[str, Any], requirements: List[str]) -> bool:
        return not any(skill not in requirements for skill in resume["skills"])
