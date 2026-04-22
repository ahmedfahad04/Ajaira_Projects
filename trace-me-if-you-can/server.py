import http.server
import socketserver

PORT = 8012
Handler = http.server.SimpleHTTPRequestHandler

print(f"Serving at http://localhost:{PORT}/index.html")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
