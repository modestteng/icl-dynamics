"""Install exact wheel bytes verified against the recorded official sources."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import urllib.parse
from html.parser import HTMLParser

root=Path(__file__).resolve().parent
wheelhouse=root/'wheelhouse'
wheelhouse.mkdir(exist_ok=True)
items=json.loads((root/'download_manifest.json').read_text())
cache=list((Path.home()/'.cache/pip').rglob('*.body'))
cache+=list(Path('/tmp').glob('pip-unpack-*/*.whl'))
by_size={}
for path in cache:
    try: by_size.setdefault(path.stat().st_size,[]).append(path)
    except FileNotFoundError: pass

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            self.urls.extend(v for k,v in attrs if k=='href')

def mirror_url(item):
    if item['url']: return item['url']
    package=item['filename'].split('-')[0].replace('_','-')
    index='https://pypi.tuna.tsinghua.edu.cn/simple/'+package+'/'
    with urllib.request.urlopen(index,timeout=60) as response:
        html=response.read().decode()
    links=Links(); links.feed(html)
    for link in links.urls:
        url=urllib.parse.urljoin(index,link)
        filename=urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit('/',1)[-1])
        if filename==item['filename']:return url
    raise RuntimeError('Exact official wheel unavailable on mirror: '+item['filename'])

for item in items:
    path=wheelhouse/item['filename']
    if path.exists() and sha(path)==item['sha256']:
        print('verified existing',path.name,flush=True)
        continue
    for candidate in by_size.get(item['size'],[]):
        try:
            if sha(candidate)==item['sha256']:
                shutil.copyfile(candidate,path)
                print('verified cached',path.name,flush=True)
                break
        except FileNotFoundError: continue
    else:
        print('download',path.name,flush=True)
        part=path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(mirror_url(item),timeout=60) as response,part.open('wb') as f:
            shutil.copyfileobj(response,f,1024*1024)
        if part.stat().st_size!=item['size'] or sha(part)!=item['sha256']:
            raise RuntimeError('Official-source checksum mismatch: '+path.name)
        part.replace(path)
        print('verified downloaded',path.name,flush=True)
requirements=root/'requirements_offline.txt'
requirements.write_text(''.join(
    f"./wheelhouse/{i['filename']} --hash=sha256:{i['sha256']}\n" for i in items))
python=root/'venv/bin/python'
subprocess.run([str(python),'-m','pip','install','--no-index','--no-deps',
                '--require-hashes','-r',str(requirements)],cwd=root,check=True)
subprocess.run([str(python),'-m','pip','check'],check=True)
with (root/'environment.txt').open('w') as f:
    subprocess.run([str(python),'-m','pip','freeze'],stdout=f,check=True)
(root/'environment_ready').write_text('Verified wheels installed; pip check passed.\n')
