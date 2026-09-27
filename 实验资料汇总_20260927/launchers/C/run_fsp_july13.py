"""Run ONE named ablation, freezing source/config/data paths before training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(variant='fsp_july13')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--data-cfg', default='dataset.yaml')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    # Set visibility before any import that could initialize CUDA.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MPLBACKEND'] = 'Agg'
    import torch
    import yaml
    from mmengine import Config
    from methods import EffB1_ZoomNeXt_FSP
    assert torch.cuda.is_available(), 'Requested CUDA device unavailable'
    cfg = Config.fromfile('configs/icod_train_fsp_july13.py')
    assert cfg.model_name == 'EffB1_ZoomNeXt_FSP'
    assert not cfg.get('load_from'), 'Ablations should start from the same encoder initialization'
    model = EffB1_ZoomNeXt_FSP(pretrained=False, **cfg.get("model_cfg", {}))
    assert all(hasattr(model, f'aim_{s}') for s in (4,3,2,1))
    assert not any(hasattr(model, n) for n in ('fde_1', 'fde_2', 'edge_head', 'sg_1', 'sg_2'))
    print('MODEL:', type(model).__name__, 'HISTORICAL FSP: AIM stages 4,3,2,1; no SG/FDE/edge/MSCA', flush=True)
    print('VISIBLE GPU:', args.gpu, torch.cuda.get_device_name(0), flush=True)
    del model
    with open(args.data_cfg, encoding='utf-8') as f:
        datasets = yaml.safe_load(f)
    counts = {}
    for name in cfg.train.data.names + cfg.test.data.names:
        item = datasets[name]
        item['root'] = str(Path(item['root']).expanduser().resolve())
        names = {}
        for kind in ('image', 'mask'):
            folder = Path(item['root']) / item[kind]['path']
            assert folder.is_dir(), f'Missing {folder}'
            names[kind] = {p.stem for p in folder.glob('*' + item[kind]['suffix']) if p.is_file()}
            assert names[kind], f'No {kind} files: {folder}'
        counts[name] = len(names['image'] & names['mask'])
        assert counts[name] > 0, f'No pairs: {name}'
        print('DATA:', name, 'images', len(names['image']), 'masks', len(names['mask']), 'pairs', counts[name], flush=True)
        if names['image'] != names['mask']:
            print('DATA WARNING: unmatched files; existing trainer uses the intersection. This is not proof of a valid standard split.', flush=True)
    total = sum(counts[n] for n in cfg.train.data.names)
    if total != 4040:
        print(f'DATA WARNING: {total} training pairs; inherited schedule was configured for 4040. Preserved for comparison, not corrected silently.', flush=True)
    subprocess.run([sys.executable, 'check_fsp_july13.py'], check=True)
    runs = root / 'fsp_july13_runs'
    runs.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_') + args.variant + '_', dir=runs))
    source = run / 'source'
    source.mkdir()
    for folder in ('methods', 'utils', 'configs', 'evidence'):
        shutil.copytree(root / folder, source / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('main_for_image.py', 'requirements.txt', 'run_fsp_july13.py', 'check_fsp_july13.py'):
        shutil.copy2(root / name, source / name)
    cfg.dump(str(source / 'selected_config.py'))
    (source / 'dataset.yaml').write_text(yaml.safe_dump(datasets, allow_unicode=True), encoding='utf-8')
    hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    (run / 'source.sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    (run / 'run.json').write_text(json.dumps(dict(variant=args.variant, model=cfg.model_name,
        flags=dict(aim_stages=[4,3,2,1]), gpu=args.gpu, dataset_pairs=counts,
        torch=torch.__version__), indent=2), encoding='utf-8')
    with (run / 'environment.txt').open('w') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    (root / 'last_fsp_july13_run.txt').write_text(str(run), encoding='utf-8')
    print('SNAPSHOT/RESULTS:', run, flush=True)
    cmd = [sys.executable, '-u', 'main_for_image.py', '--config', 'selected_config.py',
           '--model-name', cfg.model_name, '--data-cfg', 'dataset.yaml',
           '--output-dir', str(run / 'results'), '--info', 'fsp_july13_restored']
    if cfg.pretrained:
        cmd.append('--pretrained')
    subprocess.run(cmd, cwd=source, check=True)


if __name__ == '__main__':
    main()
