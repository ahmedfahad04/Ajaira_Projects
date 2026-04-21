class FitnessTracker:
    def __init__(self, height, weight, age, sex) -> None:
        self.height = height
        self.weight = weight
        self.age = age
        self.sex = sex
        self.BMI_std = [
            {"male": [20, 25]},
            {"female": [19, 24]}
        ]

    def get_BMI(self):
        return self.weight / self.height ** 2

    def _get_bmi_bounds(self):
        """Extract BMI bounds based on sex"""
        for standard in self.BMI_std:
            if self.sex in standard:
                return standard[self.sex]
        raise ValueError(f"Unknown sex: {self.sex}")

    def condition_judge(self):
        BMI = self.get_BMI()
        lower_bound, upper_bound = self._get_bmi_bounds()
        
        return (1 if BMI > upper_bound else
                -1 if BMI < lower_bound else 0)

    def _calculate_bmr(self):
        """Calculate Basal Metabolic Rate"""
        base = 10 * self.weight + 6.25 * self.height - 5 * self.age
        return base + (5 if self.sex == "male" else -161)

    def _get_activity_factor(self, condition):
        """Get activity factor based on body condition"""
        factors = {1: 1.2, -1: 1.6, 0: 1.4}
        return factors[condition]

    def calculate_calorie_intake(self):
        BMR = self._calculate_bmr()
        condition = self.condition_judge()
        activity_factor = self._get_activity_factor(condition)
        return BMR * activity_factor
