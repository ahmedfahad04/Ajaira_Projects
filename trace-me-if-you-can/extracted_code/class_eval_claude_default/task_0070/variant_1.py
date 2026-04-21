class PersonRequest:
    VALID_SEXES = {"Man", "Woman", "UGM"}
    MAX_NAME_LENGTH = 33
    PHONE_NUMBER_LENGTH = 11
    
    def __init__(self, name: str, sex: str, phoneNumber: str):
        self.name = self._validate_name(name)
        self.sex = self._validate_sex(sex)
        self.phoneNumber = self._validate_phoneNumber(phoneNumber)

    def _validate_name(self, name: str) -> str:
        return name if name and len(name) <= self.MAX_NAME_LENGTH else None

    def _validate_sex(self, sex: str) -> str:
        return sex if sex in self.VALID_SEXES else None

    def _validate_phoneNumber(self, phoneNumber: str) -> str:
        return phoneNumber if phoneNumber and len(phoneNumber) == self.PHONE_NUMBER_LENGTH and phoneNumber.isdigit() else None
