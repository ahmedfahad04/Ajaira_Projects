class IndividualRequest:
    def __init__(self, individual_name: str, gender: str, contact_number: str):
        self.individual_name = self.__validate_name(individual_name)
        self.gender = self.__validate_gender(gender)
        self.contact_number = self.__validate_contact_number(contact_number)

    def __validate_name(self, individual_name: str) -> str:
        if not individual_name:
            return None
        if len(individual_name) > 33:
            return None
        return individual_name

    def __validate_gender(self, gender: str) -> str:
        valid_genders = ["Man", "Woman", "UGM"]
        if gender not in valid_genders:
            return None
        return gender

    def __validate_contact_number(self, contact_number: str) -> str:
        if not contact_number:
            return None
        if len(contact_number) != 11 or not contact_number.isdigit():
            return None
        return contact_number
