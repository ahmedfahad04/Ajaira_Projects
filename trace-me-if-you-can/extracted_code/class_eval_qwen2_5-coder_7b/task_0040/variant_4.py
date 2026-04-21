class HealthProfile:
    def __init__(self, h, w, a, s) -> None:
        self.height = h
        self.weight = w
        self.age = a
        self.sex = s
        self.bmi_ranges = [
            {"male": [20, 25]},
            {"female": [19, 24]}
        ]

    def compute_bmi(self):
        return self.weight / (self.height ** 2)

    def classify_bmi(self):
        bmi = self.compute_bmi()
        ranges = self.bmi_ranges[0]["male"] if self.sex == "male" else self.bmi_ranges[1]["female"]
        if bmi > ranges[1]:
            return "overweight"
        elif bmi < ranges[0]:
            return "underweight"
        else:
            return "normal"

    def estimate_daily_calories(self):
        if self.sex == "male":
            daily_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            daily_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        classification = self.classify_bmi()
        if classification == "overweight":
            multiplier = 1.2
        elif classification == "underweight":
            multiplier = 1.6
        else:
            multiplier = 1.4
        return daily_calories * multiplier
