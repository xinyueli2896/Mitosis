# Known issues

## UNRESOLVED: jigs108.mid (Nottingham test input) sounds off the beat

Reported 2026-08-24, tabled. What is established (job 178939,
`check_beat_alignment.sbatch`):

- **Ruled out: quantization smear.** All four `input/not_split` test
  songs are perfectly on the 16th grid (0.0% off-grid onsets, max
  deviation 0.000) in both melody and chord files. The triplet-encoding
  hypothesis is refuted for these files (though `ashover2.mid` in
  `input/` shows real 1/3-step triplet deviations, so the failure mode
  exists elsewhere in the corpus family).
- **Confirmed but not yet tied to the symptom: none of the test songs
  are 4/4.** jigs108 is 6/8 (12 steps/bar), waltzes30/5 are 3/4 (12),
  ashover28 is 2/4 (8). Every 16-frames-per-bar assumption mis-slices
  them: `PROMPT_LENGTH=64` cuts mid-bar (5⅓ bars of a 12-step tune),
  `eval_metrics.FRAMES_PER_BAR=16` bins "within-bar position" wrongly,
  structure metrics slice wrong bars. None of the files carry tempo
  events (120 BPM default applies uniformly — consistent, so harmless).

Leading hypotheses for the audible symptom, untested:

1. Mid-bar prompt cuts (if what was heard was a prompt render or
   continuation): a seam at frame 64 lands mid-bar for all four songs
   and audibly breaks the beat.
2. Compound-pulse phase: chords on-grid but placed against quarter-note
   logic rather than the 6/8 dotted-quarter pulse.
3. Pickup/anacrusis at t=0 shifting the perceived barline.

Cheap next steps when picked up: use `PROMPT_LENGTH=96` for Nottingham
runs (96 is a multiple of 12 and 16, so prompts end on a barline for
every meter in the test set) and re-listen; extend
`check_beat_alignment.py` with a bar-phase histogram (chord onsets
modulo the file's own bar length) and pickup detection to decide
between hypotheses 2 and 3.

Not affected: the C.1 reconstruction study (frame-indexed, no bar
assumptions) and Nottingham training data integrity (notes are
on-grid).
