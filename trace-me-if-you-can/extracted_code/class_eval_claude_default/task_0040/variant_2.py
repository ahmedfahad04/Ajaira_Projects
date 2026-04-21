from enum import Enum

class BodyCondition(Enum):
    OVERWEIGHT = 1
    UNDERWEIGHT = -1
    NORMAL = 0

class FitnessTracker:
    def __init__(self, height, weight, age, sex) -> None:
        self.height = height
        self.weight = weight
        self.age = age
        self.sex = sex
        self._bmi_cache = None
        self._condition_cache = None

    @property
    def bmi_range(self):
        return (20, 25) if self.sex == "male" else (19, 24)

    def get_BMI(self):
        if self._bmi_cache is None:
            self._bmi_cache = self.weight / self.height ** 2
        return self._bmi_cache

    def condition_judge(self):
        if self._condition_cache is None:
            BMI = self.get_BMI()
            min_bmi, max_bmi = self.bmi_range
            
            if BMI > max_bmi:
                self._condition_cache = BodyCondition.OVERWEIGHT.value
            elif BMI < min_bmi:
                self._condition_cache = BodyCondition.UNDERWEIGHT.value
            else:
                self._condition_cache = BodyCondition.NORMAL.value
        
        return self._condition_cache

    def calculate_calorie_intake(self):
        sex_modifier = 5 if self.sex == "male" else -161
        BMR = 10 * self.weight + 6.25 * self.height - 5 * self.age + sex_modifier
        
        condition = self.condition_judge()
        if condition == BodyCondition.OVERWEIGHT.value:
            return BMR * 1.2
        elif condition == BodyCondition.UNDERWEIGHT.value:
            return BMR * 1.6
        else:
            return BMR * 1.4
