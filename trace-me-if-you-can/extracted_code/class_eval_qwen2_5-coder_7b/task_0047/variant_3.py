class IPAddressChecker:
    def __init__(self, ip):
        self.ip = ip

    def is_ip_valid(self):
        sections = self.ip.split('.')
        if len(sections) != 4:
            return False
        for section in sections:
            if not section.isdigit() or not 0 <= int(section) <= 255:
                return False
        return True

    def get_sections(self):
        if self.is_ip_valid():
            return self.ip.split('.')
        else:
            return []

    def binary_representation(self):
        if self.is_ip_valid():
            binary_sections = [f'{int(section):08b}' for section in self.get_sections()]
            return '.'.join(binary_sections)
        else:
            return ''
