class UserInput:
    def __init__(self, person_name: str, biological_sex: str, phone: str):
        self.person_name = self.check_name(person_name)
        self.biological_sex = self.sex_check(biological_sex)
        self.phone = self.validate_phone(phone)

    def check_name(self, person_name: str) -> str:
        if not person_name:
            return None
        if len(person_name) > 33:
            return None
        return person_name

    def sex_check(self, biological_sex: str) -> str:
        sexes = ["Man", "Woman", "UGM"]
        if biological_sex not in sexes:
            return None
        return biological_sex

    def validate_phone(self, phone: str) -> str:
        if not phone:
            return None
        if len(phone) != 11 or not phone.isdigit():
            return None
        return phone
