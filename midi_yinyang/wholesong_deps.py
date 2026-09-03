"""Dependency preflight for the whole-song-gen external baseline.

Their repo ships no requirements file, and its inference path imports a
long tail of packages at MODULE scope -- including several that only
training uses (torchvision via losses/lpips.py, omegaconf/requests via
losses/util.py). Discovering them one failed 12h GPU job at a time is
the failure mode this script exists to prevent: it probes the exact
import chain both drivers use, installs whatever is missing, and
repeats until the chain imports clean or an unknown module appears.

TORCH IS PROTECTED. torch-family wheels (torchvision/torchaudio) are
installed with --no-deps and pinned to the pairing for the venv's own
torch, and the torch version is verified unchanged at the end -- a
plain `pip install torchvision` would happily swap the CUDA build out
from under the models.

Run from the whole_song_gen repo root (both sbatch wrappers do):
    python wholesong_deps.py
"""

import re
import subprocess
import sys

# import name -> pip name, for everything a static scan of the repo
# found outside the standard library. Reachability on the inference
# path is noted; unreachable ones are listed so a future import error
# resolves without another scan.
PIP_NAME = {
    'mir_eval': 'mir_eval',            # data_utils/utils/read_file.py
    'sklearn': 'scikit-learn',         # utils/format_converter.py
    'networkx': 'networkx',            # tonal_reduction_algo/main.py
    'matplotlib': 'matplotlib',        # tonal_reduction_algo/main.py
    'omegaconf': 'omegaconf',          # losses/util.py (via lpips)
    'requests': 'requests',            # losses/util.py (via lpips)
    'tqdm': 'tqdm',
    'pretty_midi': 'pretty_midi',
    'numpy': 'numpy',
    'PIL': 'pillow',                   # stable_diffusion/util.py
    'labml': 'labml',                  # stable_diffusion/util.py (orphan)
    'torchvision': None,               # handled by torch pairing below
    # flash_attn is imported inside a try/ except in unet_attention.py --
    # optional by construction, never installed here.
}

# torchvision wheel paired to each torch minor; --no-deps keeps pip from
# touching torch itself.
TORCHVISION_FOR_TORCH = {
    '2.0': '0.15.2', '2.1': '0.16.2', '2.2': '0.17.2', '2.3': '0.18.1',
    '2.4': '0.19.1', '2.5': '0.20.1', '2.6': '0.21.0', '2.7': '0.22.0',
    '2.8': '0.23.0',
}

PROBE = '''
import data_utils.read_pop909_data
import data_utils.pytorch_datasets.counterpoint_dataset
import data_utils.pytorch_datasets.leadsheet_dataset
import data_utils.midi_output
import inference.generation_operations
import experiments.whole_song_gen
import params
print("PROBE-OK")
'''


def torch_version():
    out = subprocess.run(
        [sys.executable, '-c', 'import torch; print(torch.__version__)'],
        capture_output=True, text=True)
    return out.stdout.strip()


def pip_install(args):
    print(f'[deps] pip install {" ".join(args)}', flush=True)
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + args)
    return r.returncode == 0


def install_torchvision():
    tv = torch_version().split('+')[0]
    minor = '.'.join(tv.split('.')[:2])
    ver = TORCHVISION_FOR_TORCH.get(minor)
    if not ver:
        print(f'[deps] FATAL: no torchvision pairing known for torch {tv}; '
              f'add it to TORCHVISION_FOR_TORCH or install manually.')
        return False
    return pip_install(['--no-deps', f'torchvision=={ver}', 'pillow'])


def main():
    torch_before = torch_version()
    print(f'[deps] torch {torch_before} (must be unchanged at exit)')

    for attempt in range(12):
        probe = subprocess.run([sys.executable, '-c', PROBE],
                               capture_output=True, text=True)
        if 'PROBE-OK' in probe.stdout:
            print(f'[deps] import chain clean after {attempt} install(s)')
            break
        err = probe.stderr
        m = re.search(r"No module named '([A-Za-z0-9_.]+)'", err)
        if not m:
            print('[deps] the probe failed for a NON-import reason; '
                  'full stderr follows:')
            print(err)
            sys.exit(1)
        mod = m.group(1).split('.')[0]
        print(f'[deps] missing: {mod}')
        if mod == 'torchvision':
            ok = install_torchvision()
        elif mod in PIP_NAME and PIP_NAME[mod]:
            ok = pip_install([PIP_NAME[mod]])
        else:
            print(f'[deps] FATAL: no pip name known for {mod!r}. Add it to '
                  f'PIP_NAME in wholesong_deps.py (check whether the '
                  f'import is guarded upstream first -- flash_attn is).')
            sys.exit(1)
        if not ok:
            print(f'[deps] FATAL: install of {mod!r} failed.')
            sys.exit(1)
    else:
        print('[deps] FATAL: still unresolved after 12 installs.')
        print(probe.stderr)
        sys.exit(1)

    torch_after = torch_version()
    if torch_after != torch_before:
        print(f'[deps] FATAL: torch changed {torch_before} -> {torch_after}. '
              f'An install pulled a different build; reinstall '
              f'torch=={torch_before} before generating anything.')
        sys.exit(1)
    print(f'[deps] torch {torch_after} unchanged; preflight OK')


if __name__ == '__main__':
    main()
