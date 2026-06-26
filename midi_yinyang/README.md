# Mitosis — Melody/Chord (and Drum/Other, Audio/Symbolic) Joint Generation

Codebase for training and inferring joint per-modality CP-transformer
generative models on POP909 (melody+chord), LAMD (drum+other), and a
roadmap to audio+symbolic.

---

## Active model variants (current focus)

Five active variants in three conceptual groups. See `VARIANTS.md` for the
full framing and `IMPLEMENTATION_REPORT.md` for status per variant. The
older "Models — what each one is" section further down covers legacy
variants kept for reference.

All variants share: CP tokenization, single-stream pretrained backbone
warm-start, per-modality Q/K/V/O projections, RoPE inside attention,
shared MoE FFN (4 experts, top-2). They differ in **sequence layout**,
**attention mask**, and **loss**.

For `--task drumnondrum`: mod_a = drum, mod_b = nondrum.

### Group A — Co-generation (symmetric joint AR)

Both modalities are predicted from each other's past. The standard
joint-generation setting.

#### **A.1 `M2CIntraCrossAttn`** (a.k.a. **DuetAttn**) — base / reference

Interleaved `[drum_t, nondrum_t, ...]` with strict causal attention; two
SDPA passes per block (intra: same-modality; cross: other-modality).

```bash
# Train
sbatch midi_yinyang/train_intra_cross_attn.sbatch

# Inference (5 modes: co, mel2chord, chord2mel, mel_only, chord_only)
sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt \
       midi_yinyang/infer_intra_cross_attn_combined.sbatch
```

#### **A.2 `M2CDuetBlockAttn`** — fair-looking fix (symmetric same-instant coupling)

DuetAttn + appended next-frame **query slots** with bidirectional
within-frame attention. Three SDPA passes per block (intra / cross-strict /
frame-bidirectional) + two gates per modality.

```bash
# Train
sbatch midi_yinyang/train_duet_block.sbatch

# Inference (Option B: query-slot decoding)
sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt \
       midi_yinyang/infer_all_rwc.sbatch          # runs #2 and #4 on RWC prompts
```

---

### Group B — Look-ahead co-generation

#### **B.1 `M2CDuetAnticipatory`** — drum shifted ahead by k frames

Same DuetAttn architecture; drum stream is reindexed in the interleaved
input so that position `2t` contains `drum_{t+k}` (default `k = 16` ≈
1 bar). Nondrum predictions get `k` frames of **future drum context**
under standard causal AR.

```bash
# Train (default lookahead = 16 frames; override with ANTICIPATION_FRAMES)
sbatch midi_yinyang/train_duet_anticipatory.sbatch

# Tune lookahead:
sbatch --export=ALL,ANTICIPATION_FRAMES=8  midi_yinyang/train_duet_anticipatory.sbatch
sbatch --export=ALL,ANTICIPATION_FRAMES=32 midi_yinyang/train_duet_anticipatory.sbatch

# Inference: not yet wired through the helper sbatch (state-dict-compatible
# with #2, but the inference loop needs the same shift applied internally
# to interpret outputs correctly). TODO.
```

---

### Group C — Conditional generation

Drum is treated as a (rehearsal / prefix) condition; the model is biased
toward "given drum, generate nondrum."

#### **C.1 `M2CDuetRehearsal`** — drum prefix + interleaved AR suffix

Sequence (length 3T): `[drum_0..drum_{T-1}]` (bidirectional within) +
standard DuetAttn-shifted interleaved suffix. Suffix runs joint AR with
full visibility into the drum prefix. Loss = CE on suffix + Brier-MSE
recon term on suffix-drum logits.

```bash
# Train (default recon_weight = 1.0)
sbatch midi_yinyang/train_duet_rehearsal.sbatch

# Tune recon weight:
sbatch --export=ALL,RECON_WEIGHT=0.5 midi_yinyang/train_duet_rehearsal.sbatch
sbatch --export=ALL,RECON_WEIGHT=0.0 midi_yinyang/train_duet_rehearsal.sbatch  # pure CE

# Inference (drum -> nondrum only; the natural use case for this variant)
sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt \
       midi_yinyang/infer_duet_rehearsal.sbatch
```

#### **C.2 `M2CDuetPrefix`** — one-way drum → nondrum (prefix-LM)

Sequence: `[drum_0..drum_{T-1}, sos_n, nondrum_0..nondrum_{T-2}]`. Drum
block bidirectional within itself; nondrum block strict causal + reads
all drum. Loss = CE on nondrum positions only. One-way drum→nondrum by
construction.

```bash
# Train
sbatch midi_yinyang/train_duet_prefix.sbatch

# Inference (drum -> nondrum only; this variant only supports that direction)
sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt \
       midi_yinyang/infer_duet_prefix.sbatch
```

---

### Common overrides for all training sbatches

```bash
# Use 2 GPUs instead of 4
sbatch --gres=gpu:2 midi_yinyang/train_<variant>.sbatch

# Switch task (melchord = POP909 mel/chord)
sbatch --export=ALL,TASK=melchord midi_yinyang/train_<variant>.sbatch

# Resume from a ckpt
sbatch --export=ALL,CKPT=ckpt/<run>/last.ckpt midi_yinyang/train_<variant>.sbatch

# Override LR / batch size
sbatch --export=ALL,MAX_LR=5e-5,BATCH_SIZE=2 midi_yinyang/train_<variant>.sbatch
```

### Variant compatibility for inference

| Variant | Inference script | Modes supported | State-dict compatible with |
|---|---|---|---|
| A.1 DuetAttn | `infer_intra_cross_attn_combined.sbatch` | co, mel2chord, chord2mel, mel_only, chord_only | — |
| A.2 DuetBlock | `infer_all_rwc.sbatch` (Option B query-slot decode) | same 5 modes | — |
| B.1 Anticipatory | TODO (state-dict-compat with A.1; just needs same shift applied) | — | A.1 |
| C.1 Rehearsal | `infer_duet_rehearsal.sbatch` | **drum → nondrum only** | — |
| C.2 Prefix | `infer_duet_prefix.sbatch` | **drum → nondrum only** | — |

---


## Repository layout

```
midi_yinyang/
├── cp_transformer.py                                     # Base CP transformer (RoFormerSymbolicTransformer)
├── cp_transformer_m2c_moe.py                             # m2c MoE (one shared backbone, dual-mask)
├── cp_transformer_m2c_moe_inference.py
├── cp_transformer_m2c_moe_combined.py
├── cp_transformer_m2c_per_layer_fusion.py                # Per-layer fusion AR
├── cp_transformer_m2c_per_layer_fusion_inference.py
├── cp_transformer_m2c_per_layer_fusion_combined.py
├── cp_transformer_m2c_mixture_head.py                    # Per-layer fusion + joint mixture head
├── cp_transformer_m2c_mixture_head_inference.py
├── cp_transformer_m2c_mixture_head_combined.py
├── cp_transformer_m2c_fusion.py                          # Two-backbones fusion (end-of-stack)
├── cp_transformer_m2c_two_backbones_crossattn.py         # *** LATEST *** two backbones + per-layer cross-attn
├── init_pretrained_into_fusion.py                        # Pretrained -> M2CFusion init
├── init_two_pretrained_into_two_backbones_crossattn.py   # Pretrained -> M2CTwoBackbonesCrossAttn init
├── make_la_drum_other.py                                 # LAMD preprocessing -> drum/other tensors
├── preprocess_large_midi_dataset.py                      # Core MIDI -> CP-token pipeline
├── xf_midi.py, settings.py                               # MIDI parsing helpers
├── transformers_roformer_moe/                            # Vendored transformers fork (MoE-capable RoFormer)
└── trail-and-error/                                      # Archived earlier variants (SameStep, mask-predict)
```

## Quick start

### 1. Install deps

```bash
pip install torch pytorch_lightning wandb pretty_midi mido six joblib numpy
```

The vendored transformers fork is picked up automatically by the
training/inference scripts via `sys.path` injection at the top of each
file. No `pip install` needed for it.

### 2. Data layout (gitignored)

```
midi_yinyang/data/
├── pop909_melody_cp4_v2.pt           # melody stream
├── pop909_melody_cp4_v2.length.pt
├── pop909_melody_cp4_v2.pitch_shift_range.pt
├── pop909_chord_cp4_v2.pt            # chord stream
├── pop909_chord_cp4_v2.length.pt
├── pop909_chord_cp4_v2.pitch_shift_range.pt
├── la_melody_cp16_v2.pt              # drum stream (named "melody" for FramedDataset compat)
├── la_melody_cp16_v2.length.pt
├── la_chord_cp16_v2.pt               # non-drum stream (named "chord" for compat)
└── la_chord_cp16_v2.length.pt

midi_yinyang/pretrained/              # gitignored; pretrained ckpts go here
midi_yinyang/ckpt/                    # gitignored; training output
midi_yinyang/input/                   # gitignored; prompt MIDIs for inference
midi_yinyang/temp/                    # gitignored; combined-output staging
```

The `FramedDataset` loader pairs a `*_chord_*.pt` file with its `*_melody_*.pt`
sibling by substituting `chord -> melody` in the filename. So to train on
LAMD-drum-vs-other, point `--path_to_dataset` at `la_chord_cp16_v2.pt` and
the loader picks up `la_melody_cp16_v2.pt` automatically (this is why the
non-drum stream is misleadingly named "chord" — zero code changes needed).

### 3. Preprocess your data (optional)

For POP909 melody+chord (uses the existing pipeline in
`preprocess_large_midi_dataset.py`'s `create_pop909_*` functions).

For LAMD drum vs. other tracks:

```bash
# Run wherever you have ~30 GB free disk for the raw LAMD MIDIs
python make_la_drum_other.py \
    --midi_root /path/to/Los-Angeles-MIDI-Dataset/MIDIs \
    --output_dir data/ \
    --max_polyphony 16 \         # match the pretrained CP transformer's training cp16
    --dedup                       # per-track note dedup
# Default: process all MIDIs (no --max_idx cap)
```

Produces `data/la_melody_cp16_v2.pt` (drums) + `data/la_chord_cp16_v2.pt`
(other) plus their `.length.pt` and `.pitch_shift_range.pt` siblings.

### 4. (Optional) Download pretrained CP transformer

```bash
mkdir -p pretrained
cd pretrained
curl -OL 'https://github.com/xinyueli2896/Mitosis/releases/download/pretrained_cp/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt'
cd ..
```

This is a `size=1` (H=768, 12 layers, `with_velocity=False`) CP transformer.
Most likely trained on `cp16` data (LAMD or similar).

---

## Models — what each one is

The "active" models below sit in `midi_yinyang/` (not under `trail-and-error/`).

| Model | File | One-line summary |
|---|---|---|
| **M2CMoE** (original) | `cp_transformer_m2c_moe.py` | One shared backbone, dual-mask (mel-keys / chord-keys passes), end-of-stack gated combination. Cross-modal info enters via masking. |
| **M2CPerLayerFusion** | `cp_transformer_m2c_per_layer_fusion.py` | One shared backbone, dual-mask per-block (per-layer fusion); cross-modal info enters at EVERY layer. AR training. |
| **M2CMixtureHead** | `cp_transformer_m2c_mixture_head.py` | M2CPerLayerFusion + K-component joint mixture head at the output. Joint sampling via posterior on `k`. |
| **M2CFusion** | `cp_transformer_m2c_fusion.py` | Two physically separate backbones (untied), end-of-stack gated combination. Adapter-only phase 1 finetuning supported. |
| **M2CTwoBackbonesCrossAttn** ← **latest** | `cp_transformer_m2c_two_backbones_crossattn.py` | Two physically separate backbones, per-layer cross-attention adapter with its own Q/K/V. Pretrained CP-ckpt loadable; two-phase training. |

Archived under `trail-and-error/`:
SameStep, M2CMaskMoE (SameStep + mask-predict), M2CPerLayerFusionMask
(per-layer fusion + mask-predict) and their inference scripts.

---

## Training commands

All training entry points share a similar CLI:
`--batch_size`, `--model_size` (small/large) or `--size` (0/1/2/3),
`--path_to_dataset`, `--checkpoint_path` (resume), `--wandb`, `--run_tag`.

### M2CMoE (baseline)

```bash
cd midi_yinyang
python cp_transformer_m2c_moe.py \
    --batch_size 8 --model_size large \
    --path_to_dataset data/pop909_chord_cp4_v2.pt \
    --moe_num_experts 4 --moe_topk 2 \
    --global_num_layers 12 \
    --wandb
```

### M2CPerLayerFusion (AR)

```bash
python cp_transformer_m2c_per_layer_fusion.py \
    --batch_size 8 --model_size large \
    --path_to_dataset data/pop909_chord_cp4_v2.pt \
    --moe_num_experts 4 --moe_topk 2 \
    --global_num_layers 12 \
    --wandb
```

### M2CMixtureHead (joint mixture)

```bash
python cp_transformer_m2c_mixture_head.py \
    --batch_size 8 --model_size large \
    --path_to_dataset data/pop909_chord_cp4_v2.pt \
    --moe_num_experts 4 --moe_topk 2 \
    --mixture_K 8 \
    --wandb
```

### M2CFusion (two backbones, end-of-stack gates)

Step 1 — init from one pretrained:

```bash
python init_pretrained_into_fusion.py \
    --pretrained pretrained/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt \
    --size 1 \
    --out pretrained/fusion_init.pt
```

Step 2 — train (Phase 1 with frozen backbone):

```bash
python cp_transformer_m2c_fusion.py \
    --batch_size 8 --model_size large \
    --path_to_dataset data/pop909_chord_cp4_v2.pt \
    --checkpoint_path pretrained/fusion_init.pt \
    --freeze_backbone \
    --moe_num_experts 4 --moe_topk 2 \
    --wandb
```

Step 3 — Phase 2 (unfreeze, joint finetune):

```bash
python cp_transformer_m2c_fusion.py \
    --batch_size 8 --model_size large \
    --path_to_dataset data/pop909_chord_cp4_v2.pt \
    --checkpoint_path ckpt/<phase1_run_name>/last.ckpt \
    --wandb
```

### M2CTwoBackbonesCrossAttn (latest) — two-phase

Step 1 — init from one duplicated pretrained ckpt:

```bash
python init_two_pretrained_into_two_backbones_crossattn.py \
    --ckpt_pretrained pretrained/cp_transformer_v0.42_size1_batch_48_schedule.epoch.00.fin.ckpt \
    --size 1 \
    --out pretrained/two_backbones_crossattn_init.pt
# By default warm-starts cross-attn adapter Q/K/V/O from backbones'
# self-attention projections. Pass --no_warm_start_adapter to disable.
```

**Phase 1** (train adapter + gates + token_type_embeddings; backbones frozen):

```bash
python cp_transformer_m2c_two_backbones_crossattn.py \
    --batch_size 8 --size 1 \
    --path_to_dataset data/pop909_chord_cp4_v2.pt \
    --checkpoint_path pretrained/two_backbones_crossattn_init.pt \
    --freeze_backbones \
    --run_tag phase1_adapter \
    --wandb
```

**Phase 2** (unfreeze everything, joint finetune):

```bash
python cp_transformer_m2c_two_backbones_crossattn.py \
    --batch_size 8 --size 1 \
    --path_to_dataset data/pop909_chord_cp4_v2.pt \
    --checkpoint_path ckpt/m2c_two_backbones_crossattn_v2.0_perlayer_sz1_phase1adapter_phase1_adapter_batch_8_schedule/last.ckpt \
    --run_tag phase2_full \
    --wandb
```

### Future: audio + symbolic with two different pretrained backbones

```bash
python init_two_pretrained_into_two_backbones_crossattn.py \
    --ckpt_pretrained_m pretrained/audio_lm.pt \
    --ckpt_pretrained_c pretrained/symbolic_lm.pt \
    --size 1 --untie_local \
    --out pretrained/audio_symbolic_init.pt
```

The `--untie_local` flag gives each modality its own
`local_embedding_{m,c}` / `local_encoder_{m,c}` / `local_decoder_{m,c}`,
needed because audio and symbolic ckpts have different vocabularies.

---

## Inference

### M2CMoE / M2CPerLayerFusion / M2CMixtureHead — same 5 modes

5 modes per model:

| Mode | Behaviour |
|---|---|
| `co` | Both prompted then jointly sampled |
| `mel2chord` | Mel given throughout; chord prompted (B-prompt) then sampled |
| `chord2mel` | Symmetric |
| `mel_only` | Mel prompted then continued; chord silenced or marginalized |
| `chord_only` | Symmetric |

Single-mode:

```bash
python cp_transformer_m2c_mixture_head_inference.py \
    --mode co \
    --ckpt ckpt/<run>/last.ckpt \
    --melody input/909mel/001.mid \
    --chord  input/909cho/001.mid \
    --prompt-length 100 --gen-length 384 \
    --temperature 1.0 \
    --model-size large
```

Combined (all 5 modes per song into a single MIDI for GarageBand
inspection — 8 named tracks per song with bar-aligned starts and MIDI
markers at song-start and prompt-end boundaries):

```bash
python cp_transformer_m2c_mixture_head_combined.py \
    --ckpt ckpt/<run>/last.ckpt \
    --mel-folder input/909mel \
    --chord-folder input/909cho \
    --output temp/mixture_head_combined/all_modes.mid \
    --prompt-length 100 --gen-length 384 \
    --max-songs 10 --temperature 1.0 \
    --model-size large
```

Output: `outputs/<ckpt_name>/<mode>/<song>.mid` for single-mode;
single MIDI at `--output` for combined.

### M2CTwoBackbonesCrossAttn — inference NOT YET BUILT

Inference scripts for the latest model haven't been written yet. They
would follow the same pattern as `cp_transformer_m2c_mixture_head_*`
(thin wrappers over `cp_transformer_m2c_moe_combined.main` with
monkey-patched `load_model` and `run_mode_for_song`). Open an issue or
ask when training is verified.

---

## Naming conventions

- `cp{N}` in dataset names = `max_polyphony=N` (CP block has N polyphony
  slots × 2 features = `subseq=2N`).
- `size` schedule (mirrors `cp_transformer.py`):
  - 0: H=512, 8 heads, 1024 intermediate, 6 layers
  - 1: H=768, 12 heads, 3072 intermediate, 12 layers
  - 2: H=1024, 16 heads, 4096 intermediate, 24 layers
  - 3: H=1280, 16 heads, 5120 intermediate, 32 layers
- `gnl{N}` in checkpoint names = `global_num_layers=N` (per-layer fusion
  family overrides the size schedule).
- `K{N}` = `mixture_K=N` for the mixture-head variants.
- `_phase1adapter` = trained with `--freeze_backbones`.
- `_untiedlocal` = trained with `--untie_local`.

---

## Common gotchas

- **`FramedDataset` pairs files by `chord <-> melody` filename
  substitution**. For LAMD drum/other use the misleading `compat` naming
  (`la_melody_cp16_v2.pt` = drums, `la_chord_cp16_v2.pt` = other).
- **Pretrained CP transformer was likely trained at `cp16`**. Re-preprocess
  pop909 at `cp16` or train fresh if you need to fine-tune from this
  pretrained on pop909.
- **W&B login**: `wandb login` once per machine. Tracked under project `MusicMOE`.
- **Vendored transformers fork**: each script injects
  `midi_yinyang/transformers_roformer_moe/src` into `sys.path` at the top.
  Don't move the fork without updating the path setup.

---

## Roadmap

- Inference scripts for `M2CTwoBackbonesCrossAttn`.
- Audio + symbolic experiment with two different pretrained backbones.
- Phase 1 / Phase 2 ablations on adapter-only vs joint finetune.
- Quantitative comparison: mixture-head joint sampling vs cross-attn
  adapter coupling.
