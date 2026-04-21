class StudentRecord:
    def __init__(self, name, grade, major):
        self.name = name
        self.grade = grade
        self.major = major
        self.course_scores = {}

class AssessmentSystem:
    def __init__(self):
        self.student_records = {}

    def add_student(self, name, grade, major):
        self.student_records[name] = StudentRecord(name, grade, major)

    def add_course_score(self, name, course, score):
        record = self.student_records.get(name)
        if record:
            record.course_scores[course] = score

    def get_gpa(self, name):
        record = self.student_records.get(name)
        if not record or not record.course_scores:
            return None
        
        score_sum = 0
        score_count = 0
        for score in record.course_scores.values():
            score_sum += score
            score_count += 1
        
        return score_sum / score_count

    def get_all_students_with_fail_course(self):
        failing_students = []
        
        def check_for_failure(record):
            for score in record.course_scores.values():
                if score < 60:
                    return True
            return False
        
        for name, record in self.student_records.items():
            if check_for_failure(record):
                failing_students.append(name)
        
        return failing_students

    def get_course_average(self, course):
        total_score = 0
        student_count = 0
        
        for record in self.student_records.values():
            if course in record.course_scores:
                score = record.course_scores[course]
                if score is not None:
                    total_score += score
                    student_count += 1
        
        return total_score / student_count if student_count > 0 else None

    def get_top_student(self):
        champion = None
        champion_gpa = 0
        
        for name in self.student_records:
            current_gpa = self.get_gpa(name)
            if current_gpa and current_gpa > champion_gpa:
                champion_gpa = current_gpa
                champion = name
        
        return champion
