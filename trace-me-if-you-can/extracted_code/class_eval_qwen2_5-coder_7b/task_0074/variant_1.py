class NetworkServer:
    def __init__(self):
        self.allowed_ip_addresses = []
        self.sent_data = {}
        self.received_data = {}

    def add_allowed_ip(self, ip):
        if ip in self.allowed_ip_addresses:
            return False
        else:
            self.allowed_ip_addresses.append(ip)
            return self.allowed_ip_addresses

    def remove_allowed_ip(self, ip):
        if ip not in self.allowed_ip_addresses:
            return False
        else:
            self.allowed_ip_addresses.remove(ip)
            return self.allowed_ip_addresses

    def receive_message(self, message):
        if not isinstance(message, dict) or "ip" not in message or "data" not in message:
            return -1
        ip = message["ip"]
        data = message["data"]
        if ip not in self.allowed_ip_addresses:
            return False
        else:
            self.received_data = {"ip": ip, "data": data}
            return self.received_data["data"]

    def send_message(self, message):
        if not isinstance(message, dict) or "ip" not in message or "data" not in message:
            return "Invalid message structure"
        self.sent_data = {"ip": message["ip"], "data": message["data"]}

    def display(self, kind):
        if kind == "sent":
            return self.sent_data
        elif kind == "received":
            return self.received_data
        else:
            return False
