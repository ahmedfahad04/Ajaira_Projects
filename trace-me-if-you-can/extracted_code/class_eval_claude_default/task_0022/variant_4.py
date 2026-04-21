class ClassRegistrationSystem:
    def __init__(self):
        self.students = []
        self.students_registration_classes = {}

    def register_student(self, student):
        existing_student = next((s for s in self.students if s == student), None)
        if existing_student is None:
            self.students.append(student)
            return 1
        return 0

    def register_class(self, student_name, class_name):
        try:
            self.students_registration_classes[student_name].append(class_name)
        except KeyError:
            self.students_registration_classes[student_name] = [class_name]
        return self.students_registration_classes[student_name]

    def get_students_by_major(self, major):
        def extract_name_if_major_matches(student):
            return student["name"] if student["major"] == major else None
        
        return [name for name in map(extract_name_if_major_matches, self.students) if name is not None]

    def get_all_major(self):
        majors = set()
        for student in self.students:
            majors.add(student["major"])
        return list(majors)

    def get_most_popular_class_in_major(self, major):
        class_frequency = {}
        students_in_major = [s for s in self.students if s["major"] == major]
        
        for student in students_in_major:
            for class_name in self.students_registration_classes.get(student["name"], []):
                class_frequency[class_name] = class_frequency.get(class_name, 0) + 1
        
        return max(class_frequency.keys(), key=lambda k: class_frequency[k])
