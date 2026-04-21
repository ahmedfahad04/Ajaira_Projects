class BodyAnalysis:
    def __init__(self, h, w, a, s) -> None:
        self.height = h
        self.weight = w
        self.age = a
        self.sex = s
        self.bmi_thresholds = [
            {"male": [20, 25]},
            {"female": [19, 24]}
        ]

    def calculate_bmi(self):
        return self.weight / (self.height ** 2)

    def interpret_bmi(self):
        bmi = self.calculate_bmi()
        thresholds = self.bmi_thresholds[0]["male"] if self.sex == "male" else self.bmi_thresholds[1]["female"]
        if bmi > thresholds[1]:
            return "obese"
        elif bmi < thresholds[0]:
            return "thin"
        else:
            return "healthy"

    def project_calories(self):
        if self.sex == "male":
            base_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            base_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        interpretation = self.interpret_bmi()
        if interpretation == "obese":
            adjustment = 1.2
        elif interpretation == "thin":
            adjustment = 1.6
        else:
            adjustment = 1.4
        return base_calories * adjustment
