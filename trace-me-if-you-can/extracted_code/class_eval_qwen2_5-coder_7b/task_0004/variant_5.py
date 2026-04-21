class AcademicRegistry:
    def __init__(self):
        self.participants = {}

    def enroll_student(self, student_id, year, department):
        self.participants[student_id] = {'student_id': student_id, 'year': year, 'department': department, 'grades': {}}

    def input_exam_grade(self, student_id, course, grade):
        if student_id in self.participants:
            self.participants[student_id]['grades'][course] = grade

    def calculate_gpa(self, student_id):
        if student_id in self.participants and self.participants[student_id]['grades']:
            grades = self.participants[student_id]['grades'].values()
            return sum(grades) / len(grades)
        else:
            return None

    def identify_students_with_failing_grades(self):
        failing_students = []
        for student_id, participant in self.participants.items():
            for course, grade in participant['grades'].items():
                if grade < 60:
                    failing_students.append(student_id)
                    break
        return failing_students

    def calculate_subject_average(self, subject):
        total_grade = 0
        grade_count = 0
        for participant in self.participants.values():
            if subject in participant['grades']:
                grade = participant['grades'][subject]
                if grade is not None:
                    total_grade += grade
                    grade_count += 1
        return total_grade / grade_count if grade_count > 0 else None

    def determine_top_student(self):
        top_student = None
        highest_gpa = 0
        for student_id, participant in self.participants.items():
            gpa = self.calculate_gpa(student_id)
            if gpa is not None and gpa > highest_gpa:
                highest_gpa = gpa
                top_student = student_id
        return top_student
