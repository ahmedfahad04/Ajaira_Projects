class HealthMonitor:
    def __init__(self, h, w, a, g) -> None:
        self.height = h
        self.weight = w
        self.age = a
        self.gender = g
        self.bmi_limits = [
            {"male": [20, 25]},
            {"female": [19, 24]}
        ]

    def compute_bmi(self):
        return self.weight / (self.height ** 2)

    def evaluate_condition(self):
        bmi = self.compute_bmi()
        limits = self.bmi_limits[0]["male"] if self.gender == "male" else self.bmi_limits[1]["female"]
        if bmi > limits[1]:
            return "overweight"
        elif bmi < limits[0]:
            return "underweight"
        else:
            return "normal"

    def estimate_calories(self):
        if self.gender == "male":
            base_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            base_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        condition = self.evaluate_condition()
        if condition == "overweight":
            multiplier = 1.2
        elif condition == "underweight":
            multiplier = 1.6
        else:
            multiplier = 1.4
        return base_calories * multiplier
