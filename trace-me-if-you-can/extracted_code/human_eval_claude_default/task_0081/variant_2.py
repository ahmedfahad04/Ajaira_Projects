grade_mapping = [
    (4.0, "A+", lambda x: x == 4.0),
    (3.7, "A", lambda x: x > 3.7),
    (3.3, "A-", lambda x: x > 3.3),
    (3.0, "B+", lambda x: x > 3.0),
    (2.7, "B", lambda x: x > 2.7),
    (2.3, "B-", lambda x: x > 2.3),
    (2.0, "C+", lambda x: x > 2.0),
    (1.7, "C", lambda x: x > 1.7),
    (1.3, "C-", lambda x: x > 1.3),
    (1.0, "D+", lambda x: x > 1.0),
    (0.7, "D", lambda x: x > 0.7),
    (0.0, "D-", lambda x: x > 0.0),
    (float('-inf'), "E", lambda x: True)
]

letter_grade = []
for gpa in grades:
    for threshold, grade, condition in grade_mapping:
        if condition(gpa):
            letter_grade.append(grade)
            break
return letter_grade
