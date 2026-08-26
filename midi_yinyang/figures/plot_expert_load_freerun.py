"""Paper figure: FREE-RUNNING per-expert top-1 load heatmaps -- the
companion to expert_load (teacher-forced). Same encoding: sequential
colormap anchored at zero, so a dead expert would be a white cell.

Measured during actual generation (MODE=co, both streams sampled, 3
held-out POP909 prompt songs, 384 frames, K+1=5 refinement rounds):
every MoE routing decision was accumulated via MOE_ROUTING_STATS, and
plotted here is the FRONTIER statistic -- the last 4 positions of each
forward (the newest committed frame pair + the two query slots), i.e.
the routing of tokens at the moment they are generated, on
self-generated context. 19,200 routed decisions per layer; the
all-decisions statistic (2.16M per layer) agrees: zero dead experts in
both models. The palest cell anywhere is 0.10 (shared, L0 e2).

Data: jobs 182746 (per-modality gates) / 182747 (shared router),
results/routing_stats_18274{6,7}.json.

Run: python figures/plot_expert_load_freerun.py  (matplotlib only)
Writes figures/expert_load_freerun.pdf / .png
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# frontier top-1 loads (job logs; raw counts in the JSONs)
SHARED = [
    [.303, .403, .096, .197], [.383, .256, .212, .149],
    [.186, .277, .205, .332], [.207, .238, .425, .130],
    [.180, .321, .262, .237], [.226, .222, .252, .301],
    [.133, .215, .409, .243], [.180, .126, .317, .377],
    [.226, .214, .260, .300], [.157, .267, .183, .394],
    [.271, .189, .287, .253], [.125, .270, .432, .173]]
MG = [
    [.211, .303, .232, .254], [.262, .145, .443, .149],
    [.170, .253, .332, .245], [.221, .294, .319, .166],
    [.258, .217, .338, .186], [.238, .211, .395, .156],
    [.216, .203, .336, .246], [.340, .295, .157, .207],
    [.236, .282, .147, .335], [.217, .178, .266, .340],
    [.206, .320, .283, .191], [.376, .166, .256, .201]]

PANELS = [('Shared router', SHARED), ('Per-modality gates', MG)]
VMAX = 0.45


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 2.9), sharey=True)

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

    cbar = fig.colorbar(im, ax=axes, fraction=0.05, pad=0.03,
                        ticks=[0, 0.1, 0.2, 0.25, 0.3, 0.4])
    cbar.set_label('top-1 load (free-running)', fontsize=7)
    cbar.ax.set_yticklabels(['0 (dead)', '.1', '.2',
                             r'.25 $=1/E$', '.3', '.4'])
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.ax.hlines(0.25, 0, 1, transform=cbar.ax.get_yaxis_transform(),
                   color='black', linewidth=0.8, linestyle=(0, (3, 2)))
    cbar.outline.set_linewidth(0.6)

    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'expert_load_freerun.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('wrote figures/expert_load_freerun.{pdf,png}')


if __name__ == '__main__':
    main()
