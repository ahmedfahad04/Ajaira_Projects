import bisect

thresholds = [0.0, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0]
grade_letters = ["E", "D-", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]

letter_grade = []
for gpa in grades:
    if gpa == 4.0:
        letter_grade.append("A+")
    else:
        index = bisect.bisect_right(thresholds, gpa)
        letter_grade.append(grade_letters[index])
return letter_grade
