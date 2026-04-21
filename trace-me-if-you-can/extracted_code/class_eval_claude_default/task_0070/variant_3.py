class PersonRequest:
    def __init__(self, name: str, sex: str, phoneNumber: str):
        validation_rules = [
            (name, self._is_valid_name, 'name'),
            (sex, self._is_valid_sex, 'sex'),
            (phoneNumber, self._is_valid_phoneNumber, 'phoneNumber')
        ]
        
        for value, validator, attr_name in validation_rules:
            setattr(self, attr_name, value if validator(value) else None)

    def _is_valid_name(self, name: str) -> bool:
        return bool(name) and len(name) <= 33

    def _is_valid_sex(self, sex: str) -> bool:
        return sex in ["Man", "Woman", "UGM"]

    def _is_valid_phoneNumber(self, phoneNumber: str) -> bool:
        return bool(phoneNumber) and len(phoneNumber) == 11 and phoneNumber.isdigit()
