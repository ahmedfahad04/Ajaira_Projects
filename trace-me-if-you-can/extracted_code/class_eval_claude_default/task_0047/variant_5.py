class IPAddress:
    def __init__(self, ip_address):
        self.ip_address = ip_address

    def is_valid(self):
        components = self.ip_address.split('.')
        
        # Check count
        if len(components) != 4:
            return False
            
        # Validate each component
        for component in components:
            # Check if numeric
            if not component.isdigit():
                return False
            # Check range
            octet_value = int(component)
            if octet_value < 0 or octet_value > 255:
                return False
                
        return True

    def get_octets(self):
        return self.ip_address.split('.') if self.is_valid() else []

    def get_binary(self):
        if not self.is_valid():
            return ''
        
        binary_representation = []
        for octet_str in self.ip_address.split('.'):
            octet_int = int(octet_str)
            binary_octet = ''
            # Manual binary conversion
            for i in range(7, -1, -1):
                if octet_int >= (1 << i):
                    binary_octet += '1'
                    octet_int -= (1 << i)
                else:
                    binary_octet += '0'
            binary_representation.append(binary_octet)
            
        return '.'.join(binary_representation)
