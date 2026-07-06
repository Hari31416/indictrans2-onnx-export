#!/usr/bin/env python3
import http.server
import json
import os
import sys
from pathlib import Path

PORT = 8010
REPO_ROOT = Path(__file__).resolve().parent.parent

class BenchmarkRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/save-benchmarks":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # Write to fixtures/live-browser-benchmarks.json
                fixtures_dir = REPO_ROOT / "fixtures"
                fixtures_dir.mkdir(exist_ok=True)
                json_path = fixtures_dir / "live-browser-benchmarks.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"Saved benchmarks to {json_path}")
                
                # Write to dashboard/public/fixtures/live-browser-benchmarks.json if dashboard exists
                dash_fixtures_dir = REPO_ROOT / "dashboard" / "public" / "fixtures"
                if dash_fixtures_dir.exists():
                    dash_json_path = dash_fixtures_dir / "live-browser-benchmarks.json"
                    with open(dash_json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved benchmarks copy to {dash_json_path}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def end_headers(self):
        # Add CORS headers for simplicity
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def translate_path(self, path):
        # Override translate_path to serve files from different locations
        # If accessing /scratch/..., map to REPO_ROOT / scratch / ...
        if path.startswith("/scratch/"):
            rel_path = path[len("/scratch/"):]
            return str(REPO_ROOT / "scratch" / rel_path)
            
        # If accessing /fixtures/..., map to REPO_ROOT / fixtures / ...
        if path.startswith("/fixtures/"):
            rel_path = path[len("/fixtures/"):]
            return str(REPO_ROOT / "fixtures" / rel_path)
            
        # Default: serve from browser-lab/ directory
        # If path starts with /browser-lab/, strip it
        if path.startswith("/browser-lab/"):
            rel_path = path[len("/browser-lab/"):]
        else:
            rel_path = path.lstrip("/")
            
        return str(REPO_ROOT / "browser-lab" / rel_path)

def main():
    # Change directory to repo root so that paths resolve nicely relative to it
    os.chdir(REPO_ROOT)
    server_address = ('', PORT)
    httpd = http.server.HTTPServer(server_address, BenchmarkRequestHandler)
    print(f"Serving IndicTrans2 ONNX Browser Lab & Benchmarks at http://127.0.0.1:{PORT}")
    while True:
        try:
            httpd.handle_request()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            sys.exit(0)
        except Exception:
            # Ignore standard socket disconnect errors
            pass

if __name__ == "__main__":
    main()
