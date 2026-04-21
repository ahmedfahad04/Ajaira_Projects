class ClassRegistrationSystem:
    def __init__(self):
        self.students = {}
        self.students_registration_classes = {}

    def register_student(self, student):
        student_name = student["name"]
        if student_name in self.students:
            return 0
        self.students[student_name] = student["major"]
        return 1

    def register_class(self, student_name, class_name):
        if student_name not in self.students_registration_classes:
            self.students_registration_classes[student_name] = []
        self.students_registration_classes[student_name].append(class_name)
        return self.students_registration_classes[student_name]

    def get_students_by_major(self, major):
        return [name for name, student_major in self.students.items() if student_major == major]

    def get_all_major(self):
        return list(set(self.students.values()))

    def get_most_popular_class_in_major(self, major):
        class_counts = {}
        for name, student_major in self.students.items():
            if student_major == major:
                for class_name in self.students_registration_classes.get(name, []):
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return max(class_counts, key=class_counts.get)
