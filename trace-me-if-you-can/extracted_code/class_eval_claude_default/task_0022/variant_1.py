from collections import defaultdict, Counter

class ClassRegistrationSystem:
    def __init__(self):
        self.students = set()
        self.students_registration_classes = defaultdict(list)

    def register_student(self, student):
        student_tuple = (student["name"], student["major"])
        if student_tuple in self.students:
            return 0
        self.students.add(student_tuple)
        return 1

    def register_class(self, student_name, class_name):
        self.students_registration_classes[student_name].append(class_name)
        return self.students_registration_classes[student_name]

    def get_students_by_major(self, major):
        return [name for name, student_major in self.students if student_major == major]

    def get_all_major(self):
        return list({student_major for _, student_major in self.students})

    def get_most_popular_class_in_major(self, major):
        class_list = []
        for name, student_major in self.students:
            if student_major == major:
                class_list.extend(self.students_registration_classes[name])
        return Counter(class_list).most_common(1)[0][0]
