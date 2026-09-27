"""Ablation 2 only: original ZoomNeXt + frequency enhancement + edge supervision."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--data-cfg', default='dataset.yaml')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    print('Ablation 2 ONLY: guidance=False, frequency=True, edge=True', flush=True)
    subprocess.run([sys.executable, '-u', str(root / 'run_weight_guided.py'),
                    '--variant', 'fder', '--gpu', args.gpu,
                    '--data-cfg', args.data_cfg], cwd=root, check=True)


if __name__ == '__main__':
    main()
