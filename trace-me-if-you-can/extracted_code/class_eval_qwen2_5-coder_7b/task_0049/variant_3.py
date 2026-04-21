class JobBoard:
        def __init__(self):
            self.listings = []
            self.candidates = []

        def add_listing(self, title, employer, qualifications):
            listing = {"title": title, "employer": employer, "qualifications": qualifications}
            self.listings.append(listing)

        def delete_listing(self, listing):
            self.listings.remove(listing)

        def apply_candidate(self, name, talents, work_experience):
            candidate = {"name": name, "talents": talents, "work_experience": work_experience}
            self.candidates.append(candidate)

        def withdraw_application(self, candidate):
            self.candidates.remove(candidate)

        def discover_jobs(self, term):
            matching_jobs = []
            for listing in self.listings:
                if term.lower() in listing["title"].lower() or any(term.lower() in qualification.lower() for qualification in listing["qualifications"]):
                    matching_jobs.append(listing)
            return matching_jobs

        def collect_applicants(self, listing):
            interested_applicants = []
            for candidate in self.candidates:
                if self.check_eligibility(candidate, listing["qualifications"]):
                    interested_applicants.append(candidate)
            return interested_applicants

        @staticmethod
        def check_eligibility(candidate, qualifications):
            for talent in candidate["talents"]:
                if talent not in qualifications:
                    return False
            return True
