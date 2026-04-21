class WeightEvaluator:
    def __init__(self, ht, wt, ag, sx) -> None:
        self.height = ht
        self.weight = wt
        self.age = ag
        self.sex = sx
        self.bmi_bounds = [
            {"male": [20, 25]},
            {"female": [19, 24]}
        ]

    def get_bmi(self):
        return self.weight / (self.height ** 2)

    def assess_health(self):
        bmi = self.get_bmi()
        bounds = self.bmi_bounds[0]["male"] if self.sex == "male" else self.bmi_bounds[1]["female"]
        if bmi > bounds[1]:
            return "excessively fat"
        elif bmi < bounds[0]:
            return "too thin"
        else:
            return "within normal range"

    def compute_daily_calories(self):
        if self.sex == "male":
            daily_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            daily_calories = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        health_status = self.assess_health()
        if health_status == "excessively fat":
            factor = 1.2
        elif health_status == "too thin":
            factor = 1.6
        else:
            factor = 1.4
        return daily_calories * factor
