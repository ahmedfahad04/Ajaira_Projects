class ServerInstance:
    def __init__(self):
        self.whitelist = []
        self.sent_packets = {}
        self.received_packets = {}

    def add_to_whitelist(self, ip):
        if ip in self.whitelist:
            return False
        else:
            self.whitelist.append(ip)
            return self.whitelist

    def remove_from_whitelist(self, ip):
        if ip not in self.whitelist:
            return False
        else:
            self.whitelist.remove(ip)
            return self.whitelist

    def handle_receive(self, packet):
        if not isinstance(packet, dict) or "ip" not in packet or "data" not in packet:
            return -1
        ip = packet["ip"]
        data = packet["data"]
        if ip not in self.whitelist:
            return False
        else:
            self.received_packets = {"ip": ip, "data": data}
            return self.received_packets["data"]

    def handle_send(self, packet):
        if not isinstance(packet, dict) or "ip" not in packet or "data" not in packet:
            return "Invalid packet structure"
        self.sent_packets = {"ip": packet["ip"], "data": packet["data"]}

    def retrieve_info(self, info_type):
        if info_type == "sent":
            return self.sent_packets
        elif info_type == "received":
            return self.received_packets
        else:
            return False
