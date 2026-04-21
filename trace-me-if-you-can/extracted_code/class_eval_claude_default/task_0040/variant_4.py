class FitnessTracker:
    def __init__(self, height, weight, age, sex) -> None:
        self.height = height
        self.weight = weight
        self.age = age
        self.sex = sex
        
        # Functional approach using lambdas
        self._bmi_calculator = lambda h, w: w / h ** 2
        self._bmr_calculator = {
            "male": lambda w, h, a: 10 * w + 6.25 * h - 5 * a + 5,
            "female": lambda w, h, a: 10 * w + 6.25 * h - 5 * a - 161
        }
        self._condition_evaluator = lambda bmi, ranges: (
            1 if bmi > ranges[1] else (-1 if bmi < ranges[0] else 0)
        )
        
        # BMI standards as nested dict lookup
        self.BMI_std = {"male": [20, 25], "female": [19, 24]}

    def get_BMI(self):
        return self._bmi_calculator(self.height, self.weight)

    def condition_judge(self):
        BMI = self.get_BMI()
        ranges = self.BMI_std[self.sex]
        return self._condition_evaluator(BMI, ranges)

    def calculate_calorie_intake(self):
        BMR = self._bmr_calculator[self.sex](self.weight, self.height, self.age)
        
        # Map condition to multiplier using dictionary lookup
        multiplier_map = {1: 1.2, -1: 1.6, 0: 1.4}
        condition = self.condition_judge()
        
        return BMR * multiplier_map[condition]
