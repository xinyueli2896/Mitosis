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

from dataclasses import dataclass


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
        # track rendered from chord_midi.txt by build_pop909_chord_midi.py,
        # both tokenized at max_polyphony=4 (melody is monophonic; chords are
        # rendered as exactly 4 voices). See preprocess_pop909.sbatch for the
        # end-to-end pipeline.
        mod_a_path='data/pop909_melody_cp4_v2.pt',
        mod_b_path='data/pop909_chord_cp4_v2.pt',
        mod_a_default_program=24,
        mod_b_default_program=0,
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
