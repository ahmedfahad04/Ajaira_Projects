class UserProfile:
    def __init__(self, full_name: str, biological_sex: str, cell_phone: str):
        self.full_name = self.clean_name(full_name)
        self.biological_sex = self.check_gender(biological_sex)
        self.cell_phone = self.sanitize_phone(cell_phone)

    def clean_name(self, full_name: str) -> str:
        if not full_name:
            return None
        if len(full_name) > 33:
            return None
        return full_name

    def check_gender(self, biological_sex: str) -> str:
        genders = ["Man", "Woman", "UGM"]
        if biological_sex not in genders:
            return None
        return biological_sex

    def sanitize_phone(self, cell_phone: str) -> str:
        if not cell_phone:
            return None
        if len(cell_phone) != 11 or not cell_phone.isdigit():
            return None
        return cell_phone
