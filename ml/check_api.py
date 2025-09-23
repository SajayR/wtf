from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
      def do_POST(self):
          length = int(self.headers.get("Content-Length", 0))
          body = self.rfile.read(length)
          print("\n--- callback ---")
          print(self.headers)
          print(body.decode("utf-8", "replace"))
          self.send_response(200)
          self.end_headers()
          self.wfile.write(b"ok")

      def log_message(self, *args):
          return  # suppress default access logs

HTTPServer(("0.0.0.0", 9000), Handler).serve_forever()


'''
--- callback ---
Host: 127.0.0.1:9000
User-Agent: python-requests/2.32.5
Accept-Encoding: gzip, deflate
Accept: */*
Connection: keep-alive
Content-Length: 180
Content-Type: application/json


{"weights_path": "/Users/cisco/Documents/CisStuff/cancer/ml/serving/current", "model_version": "local-1758628402", "metrics": {"auc": 0.9978469135802469, "f1": 0.9749430523917996}}
'''