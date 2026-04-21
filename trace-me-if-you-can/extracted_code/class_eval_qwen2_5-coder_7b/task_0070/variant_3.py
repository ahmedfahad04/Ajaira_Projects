class ContactInfo:
    def __init__(self, owner_name: str, gender_identity: str, phone_number: str):
        self.owner_name = self.name_check(owner_name)
        self.gender_identity = self.gender_validation(gender_identity)
        self.phone_number = self.number_validation(phone_number)

    def name_check(self, owner_name: str) -> str:
        if not owner_name:
            return None
        if len(owner_name) > 33:
            return None
        return owner_name

    def gender_validation(self, gender_identity: str) -> str:
        acceptable_genders = ["Man", "Woman", "UGM"]
        if gender_identity not in acceptable_genders:
            return None
        return gender_identity

    def number_validation(self, phone_number: str) -> str:
        if not phone_number:
            return None
        if len(phone_number) != 11 or not phone_number.isdigit():
            return None
        return phone_number
