class StudentManager:
    def __init__(self):
        self.learners = {}

    def insert_student(self, identifier, rank, field):
        self.learners[identifier] = {'identifier': identifier, 'rank': rank, 'field': field, 'enrollments': {}}

    def enter_exam_result(self, identifier, subject, result):
        if identifier in self.learners:
            self.learners[identifier]['enrollments'][subject] = result

    def calculate_gpa(self, identifier):
        if identifier in self.learners and self.learners[identifier]['enrollments']:
            scores = self.learners[identifier]['enrollments'].values()
            return sum(scores) / len(scores)
        else:
            return None

    def fetch_students_with_failures(self):
        flagged_students = []
        for identifier, learner in self.learners.items():
            for subject, result in learner['enrollments'].items():
                if result < 60:
                    flagged_students.append(identifier)
                    break
        return flagged_students

    def determine_subject_average(self, subject):
        total_score = 0
        entry_count = 0
        for learner in self.learners.values():
            if subject in learner['enrollments']:
                score = learner['enrollments'][subject]
                if score is not None:
                    total_score += score
                    entry_count += 1
        return total_score / entry_count if entry_count > 0 else None

    def identify_top_learner(self):
        leading_learner = None
        highest_gpa = 0
        for identifier, learner in self.learners.items():
            gpa = self.calculate_gpa(identifier)
            if gpa is not None and gpa > highest_gpa:
                highest_gpa = gpa
                leading_learner = identifier
        return leading_learner
