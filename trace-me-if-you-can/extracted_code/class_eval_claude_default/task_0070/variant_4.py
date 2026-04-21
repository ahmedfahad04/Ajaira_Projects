class PersonRequest:
    def __init__(self, name: str, sex: str, phoneNumber: str):
        self.name = self._process_field(name, 'name')
        self.sex = self._process_field(sex, 'sex')
        self.phoneNumber = self._process_field(phoneNumber, 'phoneNumber')

    def _process_field(self, value: str, field_type: str) -> str:
        validation_map = {
            'name': lambda v: v if v and len(v) <= 33 else None,
            'sex': lambda v: v if v in ["Man", "Woman", "UGM"] else None,
            'phoneNumber': lambda v: v if v and len(v) == 11 and v.isdigit() else None
        }
        return validation_map[field_type](value)

    def _validate_name(self, name: str) -> str:
        if not name:
            return None
        if len(name) > 33:
            return None
        return name

    def _validate_sex(self, sex: str) -> str:
        if sex not in ["Man", "Woman", "UGM"]:
            return None
        return sex

    def _validate_phoneNumber(self, phoneNumber: str) -> str:
        if not phoneNumber:
            return None
        if len(phoneNumber) != 11 or not phoneNumber.isdigit():
            return None
        return phoneNumber
