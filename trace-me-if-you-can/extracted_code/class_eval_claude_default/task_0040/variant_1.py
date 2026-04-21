class FitnessTracker:
    BMI_STANDARDS = {
        "male": (20, 25),
        "female": (19, 24)
    }
    
    def __init__(self, height, weight, age, sex) -> None:
        self.height = height
        self.weight = weight
        self.age = age
        self.sex = sex

    def get_BMI(self):
        return self.weight / self.height ** 2

    def condition_judge(self):
        BMI = self.get_BMI()
        min_bmi, max_bmi = self.BMI_STANDARDS[self.sex]
        
        if BMI > max_bmi:
            return 1  # too fat
        elif BMI < min_bmi:
            return -1  # too thin
        else:
            return 0  # normal

    def calculate_calorie_intake(self):
        # Calculate BMR using Mifflin-St Jeor equation
        base_bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age
        BMR = base_bmr + (5 if self.sex == "male" else -161)
        
        condition = self.condition_judge()
        activity_multipliers = {1: 1.2, -1: 1.6, 0: 1.4}
        
        return BMR * activity_multipliers[condition]
