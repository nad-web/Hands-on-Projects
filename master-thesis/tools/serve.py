#!/usr/bin/env python3
"""Local-only, dependency-free preview. Nothing is uploaded."""
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import argparse
import functools
import threading
import webbrowser

p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8000);p.add_argument('--no-browser',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
class Handler(SimpleHTTPRequestHandler):
    extensions_map={**SimpleHTTPRequestHandler.extensions_map,'.js':'text/javascript','.json':'application/json','.webmanifest':'application/manifest+json'}
    def end_headers(self):
        self.send_header('Cache-Control','no-cache')
        super().end_headers()
try:server=ThreadingHTTPServer(('127.0.0.1',a.port),functools.partial(Handler,directory=str(root)))
except OSError as e:raise SystemExit(f'Could not start preview on port {a.port}: {e}. Try --port 8001.')
url=f'http://127.0.0.1:{a.port}/';print('Forecast Atlas is running at '+url+'\nPress Ctrl+C to stop. No files are uploaded.')
if not a.no_browser:threading.Timer(.6,lambda:webbrowser.open(url)).start()
try:server.serve_forever()
except KeyboardInterrupt:server.server_close()
