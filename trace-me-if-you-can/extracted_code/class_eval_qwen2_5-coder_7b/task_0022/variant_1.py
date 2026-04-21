class StudentRegistrationSystem:

    def __init__(self):
        self.enrolled_students = []
        self.class_enrollments = {}

    def add_student(self, student):
        if student not in self.enrolled_students:
            self.enrolled_students.append(student)
            return True
        return False

    def enroll_student_in_class(self, student_name, class_name):
        if student_name not in self.class_enrollments:
            self.class_enrollments[student_name] = [class_name]
        else:
            self.class_enrollments[student_name].append(class_name)
        return self.class_enrollments[student_name]

    def retrieve_students_by_major(self, major):
        students_in_major = [student for student in self.enrolled_students if student["major"] == major]
        return [student["name"] for student in students_in_major]

    def fetch_all_majors(self):
        unique_majors = set()
        for student in self.enrolled_students:
            unique_majors.add(student["major"])
        return list(unique_majors)

    def determine_most_popular_class_in_major(self, major):
        class_counts = {}
        for student in self.enrolled_students:
            if student["major"] == major:
                for class_name in self.class_enrollments[student["name"]]:
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
        return max(class_counts, key=class_counts.get)
