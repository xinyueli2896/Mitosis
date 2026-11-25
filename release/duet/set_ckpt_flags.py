"""Write a copy of a checkpoint with some config buffers (*_flag) set.

The decode rebuilds a model from the *_flag buffers in the checkpoint,
so a checkpoint whose flags are wrong decodes as the wrong model. Runs
warm-started before the 2026-11-25 fix took the init checkpoint's
default flags. Repair one by writing the values the run was trained with:

    python set_ckpt_flags.py <in.ckpt> <out_dir> joint_frame_flag=3 \\
        global_joint_attn_flag=2 slot_sees_prev_frame_flag=1

The copy keeps the file name (so its metric tag still resolves) and
lands in <out_dir>; the original is never touched.
"""
import os
import sys

import torch


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    src, out_dir, sets = argv[0], argv[1], argv[2:]
    ck = torch.load(src, map_location='cpu', weights_only=False)
    sd = ck['state_dict'] if 'state_dict' in ck else ck
    for kv in sets:
        k, v = kv.split('=', 1)
        if k not in sd:
            sys.exit(f'{k} is not a key of {src}')
        old = sd[k]
        sd[k] = torch.tensor(type(old.item())(float(v)), dtype=old.dtype)
        print(f'{k}: {old.tolist()} -> {sd[k].tolist()}')
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.basename(src))
    if os.path.abspath(dst) == os.path.abspath(src):
        sys.exit('out_dir must differ from the source directory')
    torch.save(ck, dst)
    print(f'wrote {dst}')


if __name__ == '__main__':
    main(sys.argv[1:])
