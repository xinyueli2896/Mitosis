"""Paper figure: per-expert top-1 load as heatmaps -- the "are there
dead experts?" figure.

One 12-layer x 4-expert grid per router design, sequential single-hue
colormap anchored at zero: a dead expert (an expert that never wins any
token's top-1) would be a near-white cell, unmistakable against the
field. Every cell is annotated with its load, so the figure doubles as
the exact utilization table. All 3 x 48 cells sit near the balanced
ideal 1/E = 0.25 (marked on the colorbar); the minimum anywhere is
0.07 (+bias, layer 1, e0) -- far from dead.

Data: per-expert top-1 load tables from analyze_moe_routing.sbatch --
  shared        job 182681 (seeded batch, SEED=0)
  + bias (mb)   job 179683
  per-modality gates (mg)  job 182680 (seeded batch, SEED=0)
Zero dead experts replicates across every batch analyzed (jobs 178410,
178528, 178945/46, 179683/84, 182634/35, 182665/66, 182680/81).

Run anywhere with matplotlib (no torch):
    python figures/plot_expert_load.py
Writes figures/expert_load.pdf (vector, for LaTeX) and .png (300 dpi).
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SHARED = [
    [.238, .323, .231, .208], [.203, .220, .286, .291],
    [.245, .306, .225, .224], [.328, .316, .190, .166],
    [.329, .147, .364, .159], [.242, .222, .184, .353],
    [.162, .169, .433, .236], [.160, .299, .279, .263],
    [.179, .271, .329, .220], [.235, .308, .252, .205],
    [.181, .253, .277, .290], [.246, .244, .238, .272]]
MB = [
    [.287, .232, .366, .115], [.065, .430, .282, .223],
    [.283, .145, .370, .201], [.184, .303, .374, .139],
    [.192, .368, .241, .199], [.368, .213, .281, .138],
    [.258, .260, .197, .284], [.195, .188, .388, .229],
    [.214, .222, .355, .209], [.152, .268, .299, .282],
    [.190, .243, .260, .308], [.235, .245, .304, .216]]
MG = [
    [.256, .248, .326, .169], [.171, .303, .164, .362],
    [.203, .269, .231, .297], [.308, .242, .264, .186],
    [.226, .229, .325, .221], [.366, .148, .227, .260],
    [.239, .207, .355, .199], [.254, .262, .281, .203],
    [.164, .260, .270, .306], [.237, .315, .226, .222],
    [.177, .347, .205, .271], [.320, .227, .247, .206]]

PANELS = [('Shared router', SHARED),
          ('Shared + modality bias', MB),
          ('Per-modality gates', MG)]
VMAX = 0.45


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
    })
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)

    for ax, (title, data) in zip(axes, PANELS):
        arr = np.array(data)
        im = ax.imshow(arr, cmap='Blues', vmin=0, vmax=VMAX,
                       aspect='auto', interpolation='nearest')
        for l in range(12):
            for e in range(4):
                v = arr[l, e]
                ax.text(e, l, f'{v:.2f}'.lstrip('0'), ha='center',
                        va='center', fontsize=6,
                        color='white' if v > 0.30 else 'black')
        ax.set_title(title, pad=4)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'e{e}' for e in range(4)])
        ax.set_xlabel('expert', labelpad=1.5)
        ax.tick_params(length=2)

    axes[0].set_yticks(range(12))
    axes[0].set_yticklabels([str(l) for l in range(12)])
    axes[0].set_ylabel('layer')

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02,
                        ticks=[0, 0.1, 0.2, 0.25, 0.3, 0.4])
    cbar.set_label('top-1 load', fontsize=7)
    cbar.ax.set_yticklabels(['0 (dead)', '.1', '.2',
                             r'.25 $=1/E$', '.3', '.4'])
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.ax.hlines(0.25, 0, 1, transform=cbar.ax.get_yaxis_transform(),
                   color='black', linewidth=0.8, linestyle=(0, (3, 2)))
    cbar.outline.set_linewidth(0.6)

    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'expert_load.{ext}'), dpi=300,
                    bbox_inches='tight')
    print('wrote figures/expert_load.{pdf,png}')


if __name__ == '__main__':
    main()
