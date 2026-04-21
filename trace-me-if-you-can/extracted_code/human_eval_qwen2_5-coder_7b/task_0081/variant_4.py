def get_letter_grade(gpa):
    return {
        4.0: "A+", 3.7: "A", 3.3: "A-", 3.0: "B+", 2.7: "B",
        2.3: "B-", 2.0: "C+", 1.7: "C", 1.3: "C-", 1.0: "D+",
        0.7: "D", 0.0: "D-"
    }.get(gpa, "E")

def convert_grades_to_letters(grades):
    letter_grade = [get_letter_grade(gpa) for gpa in grades]
    return letter_grade
