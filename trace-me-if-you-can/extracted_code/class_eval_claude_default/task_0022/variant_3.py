class ClassRegistrationSystem:
    def __init__(self):
        self.students = []
        self.students_registration_classes = {}

    def register_student(self, student):
        return int(self._add_student_if_new(student))

    def _add_student_if_new(self, student):
        if not any(s["name"] == student["name"] and s["major"] == student["major"] for s in self.students):
            self.students.append(student)
            return True
        return False

    def register_class(self, student_name, class_name):
        self.students_registration_classes.setdefault(student_name, []).append(class_name)
        return self.students_registration_classes[student_name]

    def get_students_by_major(self, major):
        return [s["name"] for s in self.students if s["major"] == major]

    def get_all_major(self):
        seen = set()
        result = []
        for student in self.students:
            if student["major"] not in seen:
                seen.add(student["major"])
                result.append(student["major"])
        return result

    def get_most_popular_class_in_major(self, major):
        from collections import Counter
        all_classes = []
        for student in filter(lambda s: s["major"] == major, self.students):
            all_classes.extend(self.students_registration_classes.get(student["name"], []))
        return Counter(all_classes).most_common(1)[0][0]
