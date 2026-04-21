class WellnessCalculator:
    def __init__(self, ht, wt, ag, sx) -> None:
        self.height = ht
        self.weight = wt
        self.age = ag
        self.sex = sx
        self.bmi_limits = [
            {"male": [20, 25]},
            {"female": [19, 24]}
        ]

    def calculate_bmi(self):
        return self.weight / (self.height ** 2)

    def evaluate_bmi(self):
        bmi = self.calculate_bmi()
        limits = self.bmi_limits[0]["male"] if self.sex == "male" else self.bmi_limits[1]["female"]
        if bmi > limits[1]:
            return "excessive weight"
        elif bmi < limits[0]:
            return "insufficient weight"
        else:
            return "ideal weight"

    def calculate_caloric_needs(self):
        if self.sex == "male":
            base_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            base_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        evaluation = self.evaluate_bmi()
        if evaluation == "excessive weight":
            factor = 1.2
        elif evaluation == "insufficient weight":
            factor = 1.6
        else:
            factor = 1.4
        return base_calories * factor
