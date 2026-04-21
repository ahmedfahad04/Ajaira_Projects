from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Student:
    name: str
    grade: str
    major: str
    courses: Dict[str, float] = field(default_factory=dict)
    
    def calculate_gpa(self) -> Optional[float]:
        if not self.courses:
            return None
        return sum(self.courses.values()) / len(self.courses)
    
    def has_failing_grade(self) -> bool:
        return any(score < 60 for score in self.courses.values())

class AssessmentSystem:
    def __init__(self):
        self.students: Dict[str, Student] = {}

    def add_student(self, name, grade, major):
        self.students[name] = Student(name, grade, major)

    def add_course_score(self, name, course, score):
        if name in self.students:
            self.students[name].courses[course] = score

    def get_gpa(self, name):
        return self.students[name].calculate_gpa() if name in self.students else None

    def get_all_students_with_fail_course(self):
        return [name for name, student in self.students.items() if student.has_failing_grade()]

    def get_course_average(self, course):
        scores = [student.courses[course] for student in self.students.values() 
                 if course in student.courses and student.courses[course] is not None]
        return sum(scores) / len(scores) if scores else None

    def get_top_student(self):
        valid_students = [(name, student.calculate_gpa()) for name, student in self.students.items() 
                         if student.calculate_gpa() is not None]
        return max(valid_students, key=lambda x: x[1])[0] if valid_students else None
