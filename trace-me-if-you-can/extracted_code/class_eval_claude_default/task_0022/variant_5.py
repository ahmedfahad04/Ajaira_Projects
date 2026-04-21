class ClassRegistrationSystem:
    def __init__(self):
        self.students = []
        self.students_registration_classes = {}

    def register_student(self, student):
        is_duplicate = self._is_student_registered(student)
        if not is_duplicate:
            self.students.append(student)
        return 0 if is_duplicate else 1

    def _is_student_registered(self, student):
        return student in self.students

    def register_class(self, student_name, class_name):
        if student_name in self.students_registration_classes:
            self.students_registration_classes[student_name] += [class_name]
        else:
            self.students_registration_classes[student_name] = [class_name]
        return self.students_registration_classes[student_name]

    def get_students_by_major(self, major):
        result = []
        for i in range(len(self.students)):
            if self.students[i]["major"] == major:
                result.append(self.students[i]["name"])
        return result

    def get_all_major(self):
        unique_majors = []
        for i in range(len(self.students)):
            current_major = self.students[i]["major"]
            if current_major not in unique_majors:
                unique_majors.append(current_major)
        return unique_majors

    def get_most_popular_class_in_major(self, major):
        class_list = []
        for i in range(len(self.students)):
            if self.students[i]["major"] == major:
                student_name = self.students[i]["name"]
                if student_name in self.students_registration_classes:
                    class_list.extend(self.students_registration_classes[student_name])
        
        max_count = 0
        most_popular = None
        unique_classes = list(set(class_list))
        
        for class_name in unique_classes:
            count = class_list.count(class_name)
            if count > max_count:
                max_count = count
                most_popular = class_name
        
        return most_popular
