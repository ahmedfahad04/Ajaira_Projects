class AcademicDatabase:
    def __init__(self):
        self.enrollees = {}

    def register_enrollee(self, student_id, level, program):
        self.enrollees[student_id] = {'student_id': student_id, 'level': level, 'program': program, 'scores': {}}

    def record_exam_score(self, student_id, course, score):
        if student_id in self.enrollees:
            self.enrollees[student_id]['scores'][course] = score

    def compute_gpa(self, student_id):
        if student_id in self.enrollees and self.enrollees[student_id]['scores']:
            scores = self.enrollees[student_id]['scores'].values()
            return sum(scores) / len(scores)
        else:
            return None

    def identify_students_with_failing_scores(self):
        failing_students = []
        for student_id, enrollee in self.enrollees.items():
            for course, score in enrollee['scores'].items():
                if score < 60:
                    failing_students.append(student_id)
                    break
        return failing_students

    def calculate_subject_average(self, subject):
        total_score = 0
        score_count = 0
        for enrollee in self.enrollees.values():
            if subject in enrollee['scores']:
                score = enrollee['scores'][subject]
                if score is not None:
                    total_score += score
                    score_count += 1
        return total_score / score_count if score_count > 0 else None

    def determine_top_enrollee(self):
        top_enrollee = None
        highest_gpa = 0
        for student_id, enrollee in self.enrollees.items():
            gpa = self.compute_gpa(student_id)
            if gpa is not None and gpa > highest_gpa:
                highest_gpa = gpa
                top_enrollee = student_id
        return top_enrollee
