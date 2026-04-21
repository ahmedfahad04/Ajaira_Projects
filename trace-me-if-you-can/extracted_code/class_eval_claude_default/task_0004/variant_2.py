from collections import defaultdict
from functools import reduce
from operator import add

class AssessmentSystem:
    def __init__(self):
        self._student_data = defaultdict(lambda: {'courses': {}})

    def add_student(self, name, grade, major):
        self._student_data[name].update({'name': name, 'grade': grade, 'major': major})

    def add_course_score(self, name, course, score):
        self._student_data[name]['courses'][course] = score

    def get_gpa(self, name):
        courses = self._student_data[name]['courses']
        return reduce(add, courses.values()) / len(courses) if courses else None

    def get_all_students_with_fail_course(self):
        def has_failing_course(student_data):
            return any(score < 60 for score in student_data['courses'].values())
        
        return list(filter(lambda name: has_failing_course(self._student_data[name]), 
                          self._student_data.keys()))

    def get_course_average(self, course):
        course_scores = [student['courses'][course] 
                        for student in self._student_data.values() 
                        if course in student['courses'] and student['courses'][course] is not None]
        return sum(course_scores) / len(course_scores) if course_scores else None

    def get_top_student(self):
        gpa_map = {name: self.get_gpa(name) for name in self._student_data.keys()}
        valid_gpas = {name: gpa for name, gpa in gpa_map.items() if gpa is not None}
        return max(valid_gpas, key=valid_gpas.get) if valid_gpas else None
