class AssessmentSystem:
    def __init__(self):
        self.students = {}

    def add_student(self, name, grade, major):
        self.students[name] = {'name': name, 'grade': grade, 'major': major, 'courses': {}}

    def add_course_score(self, name, course, score):
        if name in self.students:
            self.students[name]['courses'][course] = score

    def get_gpa(self, name):
        try:
            courses = self.students[name]['courses']
            return sum(courses.values()) / len(courses) if courses else None
        except KeyError:
            return None

    def get_all_students_with_fail_course(self):
        result = []
        for name, student in self.students.items():
            if any(score < 60 for score in student['courses'].values()):
                result.append(name)
        return result

    def get_course_average(self, course):
        valid_scores = []
        for student in self.students.values():
            if course in student['courses']:
                score = student['courses'][course]
                if score is not None:
                    valid_scores.append(score)
        
        if not valid_scores:
            return None
        
        total_score = 0
        for score in valid_scores:
            total_score += score
        return total_score / len(valid_scores)

    def get_top_student(self):
        best_student = None
        highest_gpa = -1
        
        for name in self.students:
            current_gpa = self.get_gpa(name)
            if current_gpa is not None and current_gpa > highest_gpa:
                highest_gpa = current_gpa
                best_student = name
        
        return best_student
