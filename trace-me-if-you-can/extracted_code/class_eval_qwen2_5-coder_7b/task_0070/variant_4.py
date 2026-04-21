class DataEntry:
    def __init__(self, individual: str, sex: str, phone: str):
        self.individual = self.validate_name(individual)
        self.sex = self.validate_sex(sex)
        self.phone = self.validate_phone(phone)

    def validate_name(self, individual: str) -> str:
        if not individual:
            return None
        if len(individual) > 33:
            return None
        return individual

    def validate_sex(self, sex: str) -> str:
        options = ["Man", "Woman", "UGM"]
        if sex not in options:
            return None
        return sex

    def validate_phone(self, phone: str) -> str:
        if not phone:
            return None
        if len(phone) != 11 or not phone.isdigit():
            return None
        return phone
