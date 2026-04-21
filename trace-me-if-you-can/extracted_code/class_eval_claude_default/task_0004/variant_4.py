import statistics

class AssessmentSystem:
    def __init__(self):
        self.students = {}

    def add_student(self, name, grade, major):
        self.students[name] = {'name': name, 'grade': grade, 'major': major, 'courses': {}}

    def add_course_score(self, name, course, score):
        if name in self.students:
            self.students[name]['courses'][course] = score

    def get_gpa(self, name):
        if name not in self.students:
            return None
        scores = list(self.students[name]['courses'].values())
        return statistics.mean(scores) if scores else None

    def get_all_students_with_fail_course(self):
        failing_students = set()
        
        for name, student in self.students.items():
            course_scores = student['courses'].values()
            if course_scores and min(course_scores) < 60:
                failing_students.add(name)
        
        return list(failing_students)

    def get_course_average(self, course):
        course_scores = []
        
        for student in self.students.values():
            if course in student['courses'] and student['courses'][course] is not None:
                course_scores.append(student['courses'][course])
        
        try:
            return statistics.mean(course_scores)
        except statistics.StatisticsError:
            return None

    def get_top_student(self):
        student_gpas = {}
        
        for name in self.students:
            gpa = self.get_gpa(name)
            if gpa is not None:
                student_gpas[name] = gpa
        
        if not student_gpas:
            return None
            
        return sorted(student_gpas.items(), key=lambda x: x[1], reverse=True)[0][0]
