"""Full-pipeline training test for ALL variants: fabricated on-disk dataset
-> real FramedDataset -> real Lightning fit() (2 steps + 1 val batch)
-> for A.3 additionally: checkpoint -> resume with BUMPED lr_total_steps,
   asserting the OneCycleLR schedule keeps the new total_steps (the
   on_load_checkpoint fix) and training continues past the old cap.
"""
import os, sys, types, shutil
import importlib.machinery

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, 'transformers_roformer_moe', 'src'))

SCRATCH = os.path.join('temp', 'fit_smoke')
shutil.rmtree(SCRATCH, ignore_errors=True)
os.makedirs(SCRATCH, exist_ok=True)


class _StubAnything:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return _StubAnything


for _name in ('pretty_midi', 'xf_midi', 'mido', 'six'):
    try:
        __import__(_name)
    except Exception:
        _m = _StubModule(_name)
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        sys.modules[_name] = _m

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. Fabricate a tiny paired dataset on disk (FramedDataset file contract:
#    <p>.pt [total_frames, poly*4], <p>.length.pt, <p>.pitch_shift_range.pt)
# ---------------------------------------------------------------------------
N_SONGS, FRAMES, POLY = 12, 80, 4
TARGET_LEN = 64          # window length (stand-in for TRAIN_LENGTH)


def make_song(drum):
    x = torch.zeros(FRAMES, POLY, 4, dtype=torch.int16)
    for t in range(FRAMES):
        n = torch.randint(1, 3, (1,)).item()
        for s in range(POLY):
            if s < n:
                x[t, s, 0] = 127 if drum else torch.randint(0, 120, (1,)).item()
                x[t, s, 1] = torch.randint(30, 90, (1,)).item()
                x[t, s, 2] = torch.randint(0, 4, (1,)).item()
                x[t, s, 3] = torch.randint(40, 100, (1,)).item()
            elif s == n:
                x[t, s, 0] = 254
            else:
                x[t, s, 1] = 255
    return x.view(FRAMES, POLY * 4)


def write_stream(prefix, drum):
    data = torch.cat([make_song(drum) for _ in range(N_SONGS)], dim=0)
    torch.save(data, prefix + '.pt')
    torch.save(torch.full((N_SONGS,), FRAMES, dtype=torch.long),
               prefix + '.length.pt')
    torch.save(torch.zeros(N_SONGS, 2, dtype=torch.int8),
               prefix + '.pitch_shift_range.pt')


drum_prefix = os.path.join(SCRATCH, 'fake_drum')
nondrum_prefix = os.path.join(SCRATCH, 'fake_nondrum')
write_stream(drum_prefix, drum=True)
write_stream(nondrum_prefix, drum=False)

from cp_transformer_m2c_moe import FramedDataset

def loaders(batch_size=2):
    tr = FramedDataset(nondrum_prefix + '.pt', TARGET_LEN, batch_size,
                       split='train', mel_path=drum_prefix + '.pt')
    va = FramedDataset(nondrum_prefix + '.pt', TARGET_LEN, batch_size,
                       split='val', mel_path=drum_prefix + '.pt')
    return (DataLoader(tr, batch_size=None, num_workers=0),
            DataLoader(va, batch_size=None, num_workers=0))


COMMON = dict(large=False, with_velocity=False,
              moe_num_experts=2, moe_topk=1, global_num_layers=2,
              max_lr=1e-4)

results = []


def fit_variant(label, build_fn, steps=2, lr_total_steps=10,
                 ckpt_dir=None, resume_from=None):
    net = build_fn(lr_total_steps)
    callbacks = []
    if ckpt_dir:
        callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_last=True,
                                          every_n_train_steps=steps,
                                          save_top_k=0))
    trainer = L.Trainer(
        accelerator='cpu', devices=1, max_steps=steps,
        val_check_interval=steps, limit_val_batches=1,
        check_val_every_n_epoch=None, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=bool(ckpt_dir),
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=callbacks,
    )
    tr, va = loaders()
    trainer.fit(net, tr, va, ckpt_path=resume_from)
    return net, trainer


def check_fit(label, build_fn):
    print(f'=== {label} ===')
    try:
        net, trainer = fit_variant(label, build_fn)
        assert trainer.global_step == 2, f'global_step={trainer.global_step}'
        print(f'  fit ok: global_step={trainer.global_step}')
        results.append((label, 'PASS'))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append((label, f'FAIL: {type(e).__name__}: {e}'))


from cp_transformer_m2c_intra_cross_attn import M2CIntraCrossAttn
from cp_transformer_m2c_duet_block import M2CDuetBlockAttn
from cp_transformer_m2c_duet_block_diffusion import M2CDuetBlockDiffusion
from cp_transformer_m2c_duet_anticipatory import M2CDuetAnticipatory
from cp_transformer_m2c_duet_rehearsal import M2CDuetRehearsal
from cp_transformer_m2c_duet_prefix import M2CDuetPrefix

check_fit('A.1 M2CIntraCrossAttn',
          lambda ts: M2CIntraCrossAttn(**COMMON, gate_init_bias=-10.0,
                                        lr_total_steps=ts))
check_fit('A.2 M2CDuetBlockAttn',
          lambda ts: M2CDuetBlockAttn(**COMMON, gate_init_bias=-10.0,
                                       query_loss_weight=1.0,
                                       lr_total_steps=ts))
check_fit('A.3 M2CDuetBlockDiffusion',
          lambda ts: M2CDuetBlockDiffusion(**COMMON, gate_init_bias=-10.0,
                                            query_loss_weight=1.0,
                                            diffusion_K=2, lr_total_steps=ts))
check_fit('B.1 M2CDuetAnticipatory',
          lambda ts: M2CDuetAnticipatory(**COMMON, gate_init_bias=-10.0,
                                          anticipation_frames=2,
                                          lr_total_steps=ts))
check_fit('C.1 M2CDuetRehearsal',
          lambda ts: M2CDuetRehearsal(**COMMON, gate_init_bias=-10.0,
                                       recon_weight=1.0, lr_total_steps=ts))
check_fit('C.2 M2CDuetPrefix',
          lambda ts: M2CDuetPrefix(**COMMON, gate_init_bias=-10.0,
                                    lr_total_steps=ts))

# ---------------------------------------------------------------------------
# 2. Resume test on A.3: train 2 steps at lr_total_steps=10, checkpoint,
#    resume with lr_total_steps=20 and max_steps=4. Assert the scheduler
#    keeps total_steps=20 after the load (on_load_checkpoint fix) and
#    training actually reaches step 4.
# ---------------------------------------------------------------------------
print('=== A.3 resume with bumped lr_total_steps ===')
try:
    a3 = lambda ts: M2CDuetBlockDiffusion(**COMMON, gate_init_bias=-10.0,
                                           query_loss_weight=1.0,
                                           diffusion_K=2, lr_total_steps=ts)
    ckpt_dir = os.path.join(SCRATCH, 'a3_ckpt')
    net1, tr1 = fit_variant('A.3 initial', a3, steps=2, lr_total_steps=10,
                             ckpt_dir=ckpt_dir)
    last = os.path.join(ckpt_dir, 'last.ckpt')
    assert os.path.exists(last), 'last.ckpt not written'

    net2 = a3(20)   # bumped schedule
    callbacks = [ModelCheckpoint(dirpath=ckpt_dir, save_last=True,
                                  every_n_train_steps=4, save_top_k=0)]
    trainer2 = L.Trainer(
        accelerator='cpu', devices=1, max_steps=4,
        val_check_interval=4, limit_val_batches=1,
        check_val_every_n_epoch=None, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=True,
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=callbacks,
    )
    tr, va = loaders()
    trainer2.fit(net2, tr, va, ckpt_path=last)

    sched = trainer2.lr_scheduler_configs[0].scheduler
    assert trainer2.global_step == 4, f'global_step={trainer2.global_step}'
    assert sched.total_steps == 20, (
        f'scheduler.total_steps={sched.total_steps} -- the saved 10-step '
        f'schedule CLOBBERED the bumped one; on_load_checkpoint fix broken'
    )
    assert sched.last_epoch == 4, f'scheduler.last_epoch={sched.last_epoch}'
    print(f'  resume ok: global_step=4, scheduler.total_steps=20, '
          f'last_epoch={sched.last_epoch}')
    results.append(('A.3 resume + LR-bump', 'PASS'))
except Exception as e:
    import traceback
    traceback.print_exc()
    results.append(('A.3 resume + LR-bump', f'FAIL: {type(e).__name__}: {e}'))

print('\n================ SUMMARY ================')
for label, status in results:
    print(f'{label:36s} {status}')
bad = [r for r in results if r[1] != 'PASS']
sys.exit(1 if bad else 0)
