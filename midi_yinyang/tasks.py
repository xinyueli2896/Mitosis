"""Task configuration: map user-friendly task names to per-modality
labels, dataset paths, and default programs.

Phase 1 of the modality-rename refactor. Internal code keeps the legacy
`mel` / `chord` variable names (`q_m`, `q_c`, `gate_m`, `gate_c`,
`mel_loss_weight`, `acc_loss_weight`, etc.) because renaming them would
break every existing checkpoint. What this module fixes is the
USER-FACING layer: CLI takes `--task melchord|drumnondrum`, the script
resolves paths/programs from the task, and every print / wandb tag /
model_name uses the task's display labels instead of hardcoded "mel"
or "chord".

Two slots, generic:
    mod_a = "stream that the original m2c lineage called melody"
    mod_b = "stream that the original m2c lineage called chord"

Concretely:
    melchord:    mod_a=mel  (program 24),  mod_b=chord  (program 0)
    drumnondrum: mod_a=drum (program 127), mod_b=nondrum (program 0)

Phase 2 (deferred) would rename the internal symbols.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Polyphony budget for the melchord DUET datasets. cp8 is the certified
# convention (cp_capacity_check): POP909 renders seventh chords as FIVE
# simultaneous notes (1 bass + 4 upper tones, build_pop909_chord_midi.py),
# so the former cp4 silently dropped the topmost tone -- the seventh --
# from 13.8% of chord frames (3.3% of chord notes). cp8 covers the
# observed maximum (5) with margin and half the padding of cp16.
# Override with MELCHORD_CP=4 to address the legacy cp4 datasets that
# the pre-migration checkpoints were trained on.
MELCHORD_CP = os.environ.get('MELCHORD_CP', '8')


@dataclass(frozen=True)
class TaskConfig:
    name: str
    mod_a_label: str
    mod_b_label: str
    mod_a_path: str
    mod_b_path: str
    mod_a_default_program: int
    mod_b_default_program: int


TASKS = {
    'melchord': TaskConfig(
        name='melchord',
        mod_a_label='mel',
        mod_b_label='chord',
        # POP909: melody track extracted by extract_pop909_melody.py, chord
        # track rendered from chord_midi.txt by build_pop909_chord_midi.py
        # (1 bass + up to 4 upper tones -> up to 5 simultaneous notes; see
        # the MELCHORD_CP note above). Both streams tokenized at the same
        # budget. See preprocess_pop909.sbatch for the end-to-end pipeline.
        mod_a_path=f'data/pop909_melody_cp{MELCHORD_CP}_v2.pt',
        mod_b_path=f'data/pop909_chord_cp{MELCHORD_CP}_v2.pt',
        mod_a_default_program=24,
        mod_b_default_program=0,
    ),
    'melchord_rev': TaskConfig(
        name='melchord_rev',
        mod_a_label='chord',
        mod_b_label='mel',
        # REVERSED POP909 melchord for conditional (C) models: they
        # condition on mod_a and generate mod_b, so chord->mel training
        # means chord as mod_a. Same POP909 files and same index-mod-10
        # split as 'melchord', roles swapped. Used for the IN-DOMAIN
        # chord->mel cascade stage B (the published YinYang conditional
        # is Nottingham-trained, hence out of domain on POP909).
        mod_a_path=f'data/pop909_chord_cp{MELCHORD_CP}_v2.pt',
        mod_b_path=f'data/pop909_melody_cp{MELCHORD_CP}_v2.pt',
        mod_a_default_program=0,
        mod_b_default_program=24,
    ),
    'melchord_pop1k7': TaskConfig(
        name='melchord_pop1k7',
        mod_a_label='mel',
        mod_b_label='chord',
        # Pop1K7 (AILabs.tw: 1,747 transcribed piano covers of J-anime /
        # K-pop / Western pop; src_001..src_004 of the annotated release,
        # 'test' held out). Melody and chord come from the release's own
        # NAMED tracks (the piano/accompaniment track is dropped), split
        # by split_named_tracks.py and tokenized by
        # preprocess_pop1k7_melchord.sbatch at MELCHORD_CP. Disjoint
        # from POP909 (different songs, different provenance).
        mod_a_path=f'data/pop1k7_melody_cp{MELCHORD_CP}_v2.pt',
        mod_b_path=f'data/pop1k7_chord_cp{MELCHORD_CP}_v2.pt',
        mod_a_default_program=24,
        mod_b_default_program=0,
    ),
    'melchord_pop1k7_rev': TaskConfig(
        name='melchord_pop1k7_rev',
        mod_a_label='chord',
        mod_b_label='mel',
        # REVERSED Pop1K7 for the conditional (C) models' chord->mel
        # direction. Same files, same index-mod-10 split, roles swapped.
        mod_a_path=f'data/pop1k7_chord_cp{MELCHORD_CP}_v2.pt',
        mod_b_path=f'data/pop1k7_melody_cp{MELCHORD_CP}_v2.pt',
        mod_a_default_program=0,
        mod_b_default_program=24,
    ),
    'melchord_nottingham': TaskConfig(
        name='melchord_nottingham',
        mod_a_label='mel',
        mod_b_label='chord',
        # Nottingham (folk): melody = instrument 0, rendered chord
        # accompaniment = instrument 1, tokenized at MELCHORD_CP to
        # match the POP909 melchord convention. Built by
        # preprocess_nottingham_melchord.sbatch. Primary use: the
        # domain-matched counterpart to the Nottingham-trained YinYang
        # conditional baseline (EXPERIMENTS.md §2.4).
        mod_a_path=f'data/nottingham_melody_cp{MELCHORD_CP}_v2.pt',
        mod_b_path=f'data/nottingham_chord_cp{MELCHORD_CP}_v2.pt',
        mod_a_default_program=24,
        mod_b_default_program=0,
    ),
    'melchord_pop909_nottingham': TaskConfig(
        name='melchord_pop909_nottingham',
        mod_a_label='mel',
        mod_b_label='chord',
        # POP909 + Nottingham combined via combine_melchord_datasets.py
        # (combine_pop909_nottingham.sbatch). POP909 is concatenated
        # FIRST, unchanged internal order, so its songs keep their
        # standalone absolute indices -- the EXPERIMENTS.md held-out ids
        # (001,011,...,091) stay held out under FramedDataset's
        # index-mod-10 split after combination. See that script's
        # docstring for the full argument.
        mod_a_path=f'data/melchord_pop909_nottingham_melody_cp{MELCHORD_CP}_v2.pt',
        mod_b_path=f'data/melchord_pop909_nottingham_chord_cp{MELCHORD_CP}_v2.pt',
        mod_a_default_program=24,
        mod_b_default_program=0,
    ),
    'melchord_pop909_nottingham_rev': TaskConfig(
        name='melchord_pop909_nottingham_rev',
        mod_a_label='chord',
        mod_b_label='mel',
        # REVERSED combined corpus, for the chord->mel arm of the C
        # models. Same two files as melchord_pop909_nottingham and the
        # same index-mod-10 split -- identical held-out songs -- with the
        # roles swapped, so a forward/reverse pair trained from these two
        # tasks differs ONLY in direction.
        mod_a_path=f'data/melchord_pop909_nottingham_chord_cp{MELCHORD_CP}_v2.pt',
        mod_b_path=f'data/melchord_pop909_nottingham_melody_cp{MELCHORD_CP}_v2.pt',
        mod_a_default_program=0,
        mod_b_default_program=24,
    ),
    'melchord_nottingham_rev': TaskConfig(
        name='melchord_nottingham_rev',
        mod_a_label='chord',
        mod_b_label='mel',
        # REVERSED direction for the conditional (C) models: they always
        # condition on mod_a and generate mod_b, so chord->mel training
        # means chord as mod_a. Same Nottingham data files as
        # melchord_nottingham, same %10 val split (identical held-out
        # songs), just swapped roles. Programs follow the streams
        # (chord 0, melody 24).
        mod_a_path=f'data/nottingham_chord_cp{MELCHORD_CP}_v2.pt',
        mod_b_path=f'data/nottingham_melody_cp{MELCHORD_CP}_v2.pt',
        mod_a_default_program=0,
        mod_b_default_program=24,
    ),
    'drumnondrum': TaskConfig(
        name='drumnondrum',
        mod_a_label='drum',
        mod_b_label='nondrum',
        mod_a_path='data/la_drum_cp16_v2.pt',
        mod_b_path='data/la_nondrum_cp16_v2.pt',
        mod_a_default_program=127,
        mod_b_default_program=0,
    ),
}


def get_task(name: str) -> TaskConfig:
    if name not in TASKS:
        raise SystemExit(
            f'unknown task {name!r}. Available: {sorted(TASKS)}'
        )
    return TASKS[name]
