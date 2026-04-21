class CommunicationServer:
    def __init__(self):
        self.authorized_ips = []
        self.outgoing_messages = {}
        self.incoming_messages = {}

    def include_ip(self, ip):
        if ip in self.authorized_ips:
            return False
        else:
            self.authorized_ips.append(ip)
            return self.authorized_ips

    def exclude_ip(self, ip):
        if ip not in self.authorized_ips:
            return False
        else:
            self.authorized_ips.remove(ip)
            return self.authorized_ips

    def receive(self, packet):
        if not isinstance(packet, dict) or "ip" not in packet or "message" not in packet:
            return -1
        ip = packet["ip"]
        message = packet["message"]
        if ip not in self.authorized_ips:
            return False
        else:
            self.incoming_messages = {"ip": ip, "message": message}
            return self.incoming_messages["message"]

    def transmit(self, packet):
        if not isinstance(packet, dict) or "ip" not in packet or "message" not in packet:
            return "Packet format is incorrect"
        self.outgoing_messages = {"ip": packet["ip"], "message": packet["message"]}

    def get_details(self, detail_type):
        if detail_type == "transmit":
            return self.outgoing_messages
        elif detail_type == "receive":
            return self.incoming_messages
        else:
            return False
