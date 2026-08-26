"""Scalar utilization / deadliness summary for the paper text.

Dead is binary (an expert never in any token's top-k receives no
gradient) and both settings measure a clean zero, so the graded
companion metric is the MIN-LOAD RATIO:

    ratio = mean over layers of (min_e load_e) / (1/E)
    deadliness = 1 - ratio

1.0 deadliness = a dead expert exists; 0 = even the most-starved
expert receives exactly its balanced share. Mean normalized load
entropy is printed alongside as the blunter concentration measure.

Reads the load tables from plot_expert_load.py (teacher-forced) and
plot_expert_load_freerun.py (free-running frontier), so it always
matches the figures.

Run: python figures/utilization_stats.py
"""

import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(fname, name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stats(name, data, E=4):
    a = np.array(data)
    mins = a.min(axis=1)
    ratio = mins.mean() * E
    p = a / a.sum(axis=1, keepdims=True)
    ent = (-(p * np.log(p)).sum(axis=1) / np.log(E)).mean()
    print(f'{name:<28} mean-min load {mins.mean():.3f}   '
          f'min-load ratio {ratio:.3f}   deadliness {1 - ratio:.3f}   '
          f'load entropy {ent:.3f}')


def main():
    tf = _load('plot_expert_load.py', '_tf')
    fr = _load('plot_expert_load_freerun.py', '_fr')
    print('TEACHER-FORCED (held-out val batch)')
    stats('  shared', tf.SHARED)
    stats('  + bias (mb)', tf.MB)
    stats('  per-modality gates (mg)', tf.MG)
    print('FREE-RUNNING (frontier statistic)')
    stats('  shared', fr.SHARED)
    stats('  per-modality gates (mg)', fr.MG)


if __name__ == '__main__':
    main()
