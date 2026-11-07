"""Download the public timm conversion at an immutable revision."""
import hashlib
from pathlib import Path
import requests
from common import ROOT, json_write, log

REPO = 'timm/vit_large_patch16_dinov3.lvd1689m'
REVISION = '30c1109559f65dea34316b0d4842d35c5771fe11'

def main():
    folder=ROOT/'models/dinov3';folder.mkdir(exist_ok=True)
    manifest={}
    for name in ['config.json','LICENSE.md','README.md','model.safetensors']:
        path=folder/name
        if not path.exists():
            url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
            with requests.get(url,stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                with path.with_suffix(path.suffix+'.tmp').open('wb') as f:
                    for chunk in response.iter_content(8*1024*1024):f.write(chunk)
            path.with_suffix(path.suffix+'.tmp').replace(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[name]={'bytes':path.stat().st_size,'sha256':digest}
        log(f'DINOv3 ready: {name}, {path.stat().st_size} bytes')
    json_write({'repo':REPO,'revision':REVISION,'files':manifest},folder/'manifest.json')

if __name__=='__main__':main()
