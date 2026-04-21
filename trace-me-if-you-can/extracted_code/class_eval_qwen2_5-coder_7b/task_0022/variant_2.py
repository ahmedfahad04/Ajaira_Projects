class SchoolClassManager:

    def __init__(self):
        self.active_students = []
        self.student_classes = {}

    def sign_up_student(self, student):
        if student not in self.active_students:
            self.active_students.append(student)
            return True
        return False

    def assign_student_to_class(self, student_name, class_name):
        if student_name not in self.student_classes:
            self.student_classes[student_name] = [class_name]
        else:
            self.student_classes[student_name].append(class_name)
        return self.student_classes[student_name]

    def list_students_by_major(self, major):
        students = [student for student in self.active_students if student["major"] == major]
        return [student["name"] for student in students]

    def get_unique_majors(self):
        majors = {student["major"] for student in self.active_students}
        return list(majors)

    def identify_popular_class_in_major(self, major):
        class_frequency = {}
        for student in self.active_students:
            if student["major"] == major:
                for class_name in self.student_classes[student["name"]]:
                    class_frequency[class_name] = class_frequency.get(class_name, 0) + 1
        return max(class_frequency, key=class_frequency.get)
