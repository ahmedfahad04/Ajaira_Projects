letter_grade = []
grade_boundaries = [
    (lambda x: x == 4.0, "A+"),
    (lambda x: x > 3.7, "A"),
    (lambda x: x > 3.3, "A-"),
    (lambda x: x > 3.0, "B+"),
    (lambda x: x > 2.7, "B"),
    (lambda x: x > 2.3, "B-"),
    (lambda x: x > 2.0, "C+"),
    (lambda x: x > 1.7, "C"),
    (lambda x: x > 1.3, "C-"),
    (lambda x: x > 1.0, "D+"),
    (lambda x: x > 0.7, "D"),
    (lambda x: x > 0.0, "D-"),
    (lambda x: True, "E")
]

for gpa in grades:
    letter_grade.append(next(grade for condition, grade in grade_boundaries if condition(gpa)))
return letter_grade
