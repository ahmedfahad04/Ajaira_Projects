from typing import Optional

class PersonRequest:
    def __init__(self, name: str, sex: str, phoneNumber: str):
        validators = {
            'name': self._validate_name,
            'sex': self._validate_sex,
            'phoneNumber': self._validate_phoneNumber
        }
        
        for field, validator in validators.items():
            setattr(self, field, validator(locals()[field]))

    def _validate_name(self, name: str) -> Optional[str]:
        return None if not name or len(name) > 33 else name

    def _validate_sex(self, sex: str) -> Optional[str]:
        allowed_values = ["Man", "Woman", "UGM"]
        return None if sex not in allowed_values else sex

    def _validate_phoneNumber(self, phoneNumber: str) -> Optional[str]:
        is_valid = phoneNumber and len(phoneNumber) == 11 and phoneNumber.isdigit()
        return phoneNumber if is_valid else None
