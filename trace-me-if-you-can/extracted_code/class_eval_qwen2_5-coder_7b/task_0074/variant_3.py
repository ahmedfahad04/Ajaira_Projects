class ServerHandler:
    def __init__(self):
        self.allowed_ips = []
        self.sent_info = {}
        self.received_info = {}

    def authorize_ip(self, ip):
        if ip in self.allowed_ips:
            return False
        else:
            self.allowed_ips.append(ip)
            return self.allowed_ips

    def revoke_ip(self, ip):
        if ip not in self.allowed_ips:
            return False
        else:
            self.allowed_ips.remove(ip)
            return self.allowed_ips

    def process_receive(self, info):
        if not isinstance(info, dict) or "ip" not in info or "payload" not in info:
            return -1
        ip = info["ip"]
        payload = info["payload"]
        if ip not in self.allowed_ips:
            return False
        else:
            self.received_info = {"ip": ip, "payload": payload}
            return self.received_info["payload"]

    def process_send(self, info):
        if not isinstance(info, dict) or "ip" not in info or "payload" not in info:
            return "Invalid info format"
        self.sent_info = {"ip": info["ip"], "payload": info["payload"]}

    def fetch_info(self, info_type):
        if info_type == "send":
            return self.sent_info
        elif info_type == "receive":
            return self.received_info
        else:
            return False
