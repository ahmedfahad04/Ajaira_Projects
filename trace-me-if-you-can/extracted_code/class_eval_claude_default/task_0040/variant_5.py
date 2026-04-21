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
        
        # Pre-compute values that don't change
        self._bmi_value = None
        self._condition_value = None

    def get_BMI(self):
        return self.weight / self.height ** 2

    def condition_judge(self):
        BMI = self.get_BMI()
        
        # Use iterator pattern to find matching standard
        bmi_range = next(
            (std[self.sex] for std in self.BMI_std if self.sex in std),
            None
        )
        
        if bmi_range is None:
            raise ValueError(f"No BMI standard found for sex: {self.sex}")
            
        # Chain comparison for cleaner logic
        if BMI > bmi_range[1]:
            return 1
        if BMI < bmi_range[0]:
            return -1
        return 0

    def calculate_calorie_intake(self):
        # Calculate BMR with conditional expression
        sex_adjustment = 5 if self.sex == "male" else -161
        BMR = 10 * self.weight + 6.25 * self.height - 5 * self.age + sex_adjustment
        
        # Use match-like pattern with tuple unpacking
        condition = self.condition_judge()
        calorie_factors = [(1, 1.2), (-1, 1.6), (0, 1.4)]
        
        factor = next(f for c, f in calorie_factors if c == condition)
        return BMR * factor
