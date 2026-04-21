class ExaminationSystem:
    def __init__(self):
        self.students = {}

    def add_student_profile(self, student_id, year, department):
        self.students[student_id] = {'student_id': student_id, 'year': year, 'department': department, 'scores': {}}

    def input_exam_mark(self, student_id, course, mark):
        if student_id in self.students:
            self.students[student_id]['scores'][course] = mark

    def calculate_student_gpa(self, student_id):
        if student_id in self.students and self.students[student_id]['scores']:
            marks = self.students[student_id]['scores'].values()
            return sum(marks) / len(marks)
        else:
            return None

    def identify_students_with_failing_marks(self):
        failing_students = []
        for student_id, student in self.students.items():
            for course, mark in student['scores'].items():
                if mark < 60:
                    failing_students.append(student_id)
                    break
        return failing_students

    def determine_course_average_mark(self, course):
        total_mark = 0
        mark_count = 0
        for student in self.students.values():
            if course in student['scores']:
                mark = student['scores'][course]
                if mark is not None:
                    total_mark += mark
                    mark_count += 1
        return total_mark / mark_count if mark_count > 0 else None

    def identify_top_student(self):
        top_student = None
        max_gpa = 0
        for student_id, student in self.students.items():
            gpa = self.calculate_student_gpa(student_id)
            if gpa is not None and gpa > max_gpa:
                max_gpa = gpa
                top_student = student_id
        return top_student
