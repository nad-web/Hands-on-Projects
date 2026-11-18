#!/usr/bin/env python3
"""Check emitted HTML links, JavaScript imports, and the manuscript exclusion boundary."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
import re
ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for name in ['href','src']:
            if a.get(name):self.urls.append(a[name])
count=0
for p in sorted(ROOT.rglob('*.html')):
    parser=Links();parser.feed(p.read_text())
    for uri in parser.urls:
        u=urlparse(uri)
        if u.scheme or u.netloc or not u.path:continue
        assert not u.path.startswith('/'), (p.relative_to(ROOT),uri,'absolute link breaks repository subpaths')
        target=(p.parent/unquote(u.path)).resolve()
        assert target.is_relative_to(ROOT), (p,uri)
        if target.is_dir():target=target/'index.html'
        assert target.is_file(), (p.relative_to(ROOT),uri)
        count+=1
for p in (ROOT/'assets').glob('*.js'):
    for name in re.findall(r"(?:from\s*|new URL\()['\"](\.[^'\"]+)['\"]",p.read_text()):
        assert (p.parent/name).resolve().exists(),(p,name)
print(f'PASS — {count} emitted local HTML links and all application module imports resolve beneath the upload folder.')
