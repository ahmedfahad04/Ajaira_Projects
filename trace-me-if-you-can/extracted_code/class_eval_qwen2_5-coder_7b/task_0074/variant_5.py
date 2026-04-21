class ServerConnector:
    def __init__(self):
        self.ips_allowed = []
        self.sent_items = {}
        self.received_items = {}

    def include_ip(self, ip):
        if ip in self.ips_allowed:
            return False
        else:
            self.ips_allowed.append(ip)
            return self.ips_allowed

    def exclude_ip(self, ip):
        if ip not in self.ips_allowed:
            return False
        else:
            self.ips_allowed.remove(ip)
            return self.ips_allowed

    def receive_data(self, data_packet):
        if not isinstance(data_packet, dict) or "ip" not in data_packet or "content" not in data_packet:
            return -1
        ip = data_packet["ip"]
        content = data_packet["content"]
        if ip not in self.ips_allowed:
            return False
        else:
            self.received_items = {"ip": ip, "content": content}
            return self.received_items["content"]

    def send_data(self, data_packet):
        if not isinstance(data_packet, dict) or "ip" not in data_packet or "content" not in data_packet:
            return "Invalid data format"
        self.sent_items = {"ip": data_packet["ip"], "content": data_packet["content"]}

    def display_info(self, info_type):
        if info_type == "send":
            return self.sent_items
        elif info_type == "receive":
            return self.received_items
        else:
            return False
