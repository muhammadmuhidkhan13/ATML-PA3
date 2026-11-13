"""Materialize authenticated Drive file references; record only public source URLs."""
import argparse,hashlib,json
from pathlib import Path
import requests

def download(manifest,root):
    root=Path(root).resolve();records=[]
    for item in json.loads(Path(manifest).read_text(encoding='utf-8')):
        target=(root/item['relative_path']).resolve()
        if root not in target.parents:raise ValueError('Download path escapes target')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.partial')
        if not target.exists():
            with requests.get(item['download_url'],stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as output:
                    for block in response.iter_content(1024**2):
                        if block:output.write(block)
            if temporary.stat().st_size!=item['bytes']:raise ValueError('Downloaded byte count differs')
            temporary.replace(target)
        if target.stat().st_size!=item['bytes']:raise ValueError('Existing file size differs')
        checksum=hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''):checksum.update(block)
        records.append({k:v for k,v in item.items() if k!='download_url'}|{'sha256':checksum.hexdigest()})
        (root/'download_manifest.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
        print(item['relative_path'],target.stat().st_size,checksum.hexdigest(),flush=True)
    return records

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path('ATML-PA3-Data'))
    args=parser.parse_args();download(args.manifest,args.root)
