import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--gpu',default='1')
p.add_argument('--data-cfg',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent
data=Path(a.data_cfg).resolve()
if not data.is_file(): raise SystemExit(f'Missing dataset config: {data}')
env=os.environ.copy()
env['CUDA_VISIBLE_DEVICES']=a.gpu
run=root/'all_group_runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
src=run/'source'
src.mkdir(parents=True)
for name in ('methods','utils','configs'):
    shutil.copytree(root/name,src/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('main_for_image.py','run_all_group.py','check_all_group.py','requirements.txt'):
    shutil.copy2(root/name,src/name)
shutil.copy2(data,run/'dataset.yaml')
manifest={str(f.relative_to(src)):hashlib.sha256(f.read_bytes()).hexdigest() for f in src.rglob('*') if f.is_file()}
(run/'source_sha256.json').write_text(json.dumps(manifest,indent=2))
with (run/'pip_freeze.txt').open('w') as f:
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
cmd=[sys.executable,'-u','main_for_image.py','--config','configs/icod_train_all_group.py','--model-name','EffB1_ZoomNeXt_AllGroup','--pretrained','--data-cfg',str(data),'--output-dir',str(run/'results'),'--info','rgpu_all_group_only']
(run/'command.json').write_text(json.dumps({'command':cmd,'CUDA_VISIBLE_DEVICES':a.gpu},indent=2))
print('SNAPSHOT/RESULTS:',run,flush=True)
print('MODEL: original ZoomNeXt + all-group RGPU at stages 5,4,3,2,1; no FSP/SG/FDE/edge',flush=True)
print('Original training configuration retained, including the historical scheduler milestone 101000.',flush=True)
raise SystemExit(subprocess.call(cmd,cwd=src,env=env))
