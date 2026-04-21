def convert_grades_to_letters(grades):
    letter_grade = [
        "A+" if gpa == 4.0 else
        "A" if gpa > 3.7 else
        "A-" if gpa > 3.3 else
        "B+" if gpa > 3.0 else
        "B" if gpa > 2.7 else
        "B-" if gpa > 2.3 else
        "C+" if gpa > 2.0 else
        "C" if gpa > 1.7 else
        "C-" if gpa > 1.3 else
        "D+" if gpa > 1.0 else
        "D" if gpa > 0.7 else
        "D-" if gpa > 0.0 else
        "E" for gpa in grades
    ]
    return letter_grade
