"""Paper figure: expert modality purity -- the emerged
specialists-and-integrators structure.

Two heatmaps (shared router vs per-modality gates), one shared
diverging colorbar: each cell is the share of melody among the tokens
that expert wins in that layer. Poles = per-stream specialists; the
neutral midpoint = experts serving both streams. Cells at the extremes
(<=10% or >=90%) are marked S (specialist); cells near the midpoint
(40-60%) are marked I (integrator). Under per-modality gates the
partition is crisper AND the mixed experts remain -- the
specialists+integrators structure emerged without any expert being
pre-assigned to a stream.

Data: purity tables from analyze_moe_routing.sbatch on the same seeded
batch (SEED=0): shared = job 182681, per-modality gates = job 182680.

Run: python figures/plot_expert_purity.py  (matplotlib only)
Writes figures/expert_purity.pdf / .png
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SHARED = [
    [1.4, 55.6, 92.4, 50.0], [84.7, 4.4, 44.5, 65.6],
    [6.1, 81.1, 47.1, 58.3], [81.6, 25.3, 58.0, 25.5],
    [25.6, 74.0, 80.2, 9.0], [39.0, 51.5, 6.0, 79.6],
    [44.6, 47.3, 76.3, 7.4], [40.7, 66.7, 63.4, 22.5],
    [44.9, 64.6, 62.7, 17.1], [50.3, 74.5, 28.6, 39.2],
    [11.2, 53.6, 48.6, 72.4], [97.9, 79.5, 7.4, 17.7]]
MG = [
    [3.5, 89.3, 33.1, 95.4], [16.3, 84.1, 64.4, 30.8],
    [11.2, 71.3, 18.3, 82.1], [84.4, 0.0, 87.7, 4.5],
    [17.0, 78.7, 79.6, 10.6], [56.0, 37.3, 1.1, 91.5],
    [83.2, 4.7, 80.8, 2.3], [21.0, 81.4, 63.7, 26.8],
    [19.4, 73.3, 75.5, 24.0], [22.5, 70.3, 13.2, 88.0],
    [0.0, 43.8, 63.3, 80.6], [99.6, 76.5, 2.9, 0.3]]


def annotate(ax, data):
    for l in range(12):
        for e in range(4):
            v = data[l][e]
            if v <= 10 or v >= 90:
                ax.text(e, l, 'S', ha='center', va='center',
                        fontsize=6.5, fontweight='bold', color='white')
            elif 40 <= v <= 60:
                ax.text(e, l, 'I', ha='center', va='center',
                        fontsize=6.5, fontweight='bold', color='black')


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.9), sharey=True)
    for ax, (title, data) in zip(
            axes, [('Shared router', SHARED),
                   ('Per-modality gates', MG)]):
        im = ax.imshow(np.array(data), cmap='RdBu', vmin=0, vmax=100,
                       aspect='auto', interpolation='nearest')
        annotate(ax, data)
        ax.set_title(title, pad=4)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'e{e}' for e in range(4)])
        ax.set_xlabel('expert', labelpad=1.5)
        ax.tick_params(length=2)
    axes[0].set_yticks(range(12))
    axes[0].set_yticklabels([str(l) for l in range(12)])
    axes[0].set_ylabel('layer')

    cbar = fig.colorbar(im, ax=axes, fraction=0.05, pad=0.03,
                        ticks=[0, 25, 50, 75, 100])
    cbar.set_label('% melody among won tokens', fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.outline.set_linewidth(0.6)

    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'expert_purity.{ext}'), dpi=300,
                    bbox_inches='tight')
    print('wrote figures/expert_purity.{pdf,png}')


if __name__ == '__main__':
    main()
