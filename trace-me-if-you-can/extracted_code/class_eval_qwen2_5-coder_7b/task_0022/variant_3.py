class EnrollmentSystem:

    def __init__(self):
        self.registered_students = []
        self.student_enrollments = {}

    def enrol_student(self, student):
        if student not in self.registered_students:
            self.registered_students.append(student)
            return True
        return False

    def register_student_for_class(self, student_name, class_name):
        if student_name not in self.student_enrollments:
            self.student_enrollments[student_name] = [class_name]
        else:
            self.student_enrollments[student_name].append(class_name)
        return self.student_enrollments[student_name]

    def collect_students_by_major(self, major):
        students_in_major = [student for student in self.registered_students if student["major"] == major]
        return [student["name"] for student in students_in_major]

    def retrieve_all_majors(self):
        unique_majors = set(student["major"] for student in self.registered_students)
        return list(unique_majors)

    def determine_most_common_class_in_major(self, major):
        class_occurrences = {}
        for student in self.registered_students:
            if student["major"] == major:
                for class_name in self.student_enrollments[student["name"]]:
                    class_occurrences[class_name] = class_occurrences.get(class_name, 0) + 1
        return max(class_occurrences, key=class_occurrences.get)
