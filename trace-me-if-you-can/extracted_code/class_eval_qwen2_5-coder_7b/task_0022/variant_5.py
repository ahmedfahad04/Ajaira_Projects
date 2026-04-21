class CourseManagementSystem:

    def __init__(self):
        self.enrolled_students = []
        self.student_class_map = {}

    def add_student(self, student):
        if student not in self.enrolled_students:
            self.enrolled_students.append(student)
            return 1
        return 0

    def add_class(self, student_name, class_name):
        if student_name not in self.student_class_map:
            self.student_class_map[student_name] = [class_name]
        else:
            self.student_class_map[student_name].append(class_name)
        return self.student_class_map[student_name]

    def get_students_in_major(self, major):
        return [student["name"] for student in self.enrolled_students if student["major"] == major]

    def get_all_majors(self):
        return list({student["major"] for student in self.enrolled_students})

    def find_popular_class_in_major(self, major):
        class_counts = {}
        for student in self.enrolled_students:
            if student["major"] == major:
                for class_name in self.student_class_map[student["name"]]:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return max(class_counts, key=class_counts.get)
