"""Fetch the fixed public checkpoints into the project, without remote model code."""
import concurrent.futures
from pathlib import Path
import requests
from common import ROOT, json_write, fingerprint

FILES={
 'miewid.safetensors':'https://huggingface.co/conservationxlabs/miewid-msv3/resolve/4f1d7f2b521149e5fe34bb85f377248ce9971a7d/model.safetensors',
 'megadescriptor.bin':'https://huggingface.co/BVRA/MegaDescriptor-L-384/resolve/33b3c6f4ee0c386a4126cc3dcd23843920613fa1/pytorch_model.bin',
 'mega_config.json':'https://huggingface.co/BVRA/MegaDescriptor-L-384/resolve/33b3c6f4ee0c386a4126cc3dcd23843920613fa1/config.json',
 'sam2.1_hiera_small.pt':'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
}

def download(item):
    name,url=item;path=ROOT/'models'/name;path.parent.mkdir(exist_ok=True)
    if not path.exists():
        response=requests.get(url,stream=True,timeout=(30,180));response.raise_for_status()
        tmp=path.with_suffix(path.suffix+'.partial')
        with tmp.open('wb') as f:
            for chunk in response.iter_content(4*1024*1024):f.write(chunk)
        tmp.replace(path)
    print(name,path.stat().st_size,flush=True)
    return {'name':name,'url':url,'bytes':path.stat().st_size,'sha256':fingerprint(path)}

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results=list(executor.map(download,FILES.items()))
    json_write(results,ROOT/'models'/'manifest.json')
