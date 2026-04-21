def determine_grade(gpa):
    grade_ranges = {
        (4.0, 4.0): "A+",
        (3.7, 3.99): "A",
        (3.3, 3.7): "A-",
        (3.0, 3.3): "B+",
        (2.7, 3.0): "B",
        (2.3, 2.7): "B-",
        (2.0, 2.3): "C+",
        (1.7, 2.0): "C",
        (1.3, 1.7): "C-",
        (1.0, 1.3): "D+",
        (0.7, 1.0): "D",
        (0.0, 0.7): "D-"
    }
    
    if gpa == 4.0:
        return "A+"
    
    for (low, high), grade in grade_ranges.items():
        if low < gpa <= high:
            return grade
    
    return "E"

letter_grade = list(map(determine_grade, grades))
return letter_grade
