class PersonRequest:
    def __init__(self, name: str, sex: str, phoneNumber: str):
        try:
            self.name = self._validate_name(name)
            self.sex = self._validate_sex(sex)  
            self.phoneNumber = self._validate_phoneNumber(phoneNumber)
        except (AttributeError, TypeError):
            self.name = None
            self.sex = None
            self.phoneNumber = None

    def _validate_name(self, name: str) -> str:
        conditions = [
            lambda: name,
            lambda: len(name) <= 33
        ]
        return name if all(condition() for condition in conditions) else None

    def _validate_sex(self, sex: str) -> str:
        valid_options = ["Man", "Woman", "UGM"]
        return next((s for s in valid_options if s == sex), None)

    def _validate_phoneNumber(self, phoneNumber: str) -> str:
        checks = [
            phoneNumber,
            len(phoneNumber) == 11,
            phoneNumber.isdigit()
        ]
        return phoneNumber if all(checks) else None
