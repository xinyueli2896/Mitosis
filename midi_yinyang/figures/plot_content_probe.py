"""Paper figure: within-stream content-responsiveness of routing --
where the per-modality gates' freed capacity went.

For each layer and stream, the probe splits frames into register
terciles and measures the mean pairwise L1 between the terciles' mean
routing distributions, against a 20-permutation shuffle null (95th
pct). The figure reports, per stream and router, the NUMBER of layers whose
effect exceeds the null -- content-driven-routing coverage out of 12. Melody panel: both routers near ceiling -- the
shared router's content channel already served melody. Chord panel:
the shared router clears the null in only 5/12 layers; per-modality
gates in 9/12 -- chord content no longer competes with the parity
signal and melody structure for one shared weight matrix.

Data: PROBE=within on the same seeded batch (SEED=0):
shared = job 182681, per-modality gates = job 182680.

Run: python figures/plot_content_probe.py  (matplotlib only)
Writes figures/content_probe.pdf / .png
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# (real L1, null95) per layer, register feature.
BASE_MEL = [(.184, .164), (.076, .048), (.125, .054), (.101, .082),
            (.089, .052), (.098, .065), (.106, .067), (.067, .069),
            (.042, .055), (.134, .057), (.033, .053), (.076, .044)]
MG_MEL = [(.175, .117), (.102, .069), (.102, .060), (.059, .068),
          (.056, .055), (.073, .052), (.098, .052), (.070, .059),
          (.050, .052), (.069, .061), (.096, .052), (.110, .044)]
BASE_CHD = [(.071, .048), (.057, .073), (.075, .078), (.074, .104),
            (.076, .109), (.189, .110), (.070, .073), (.092, .092),
            (.132, .072), (.114, .075), (.078, .066), (.041, .067)]
MG_CHD = [(.078, .049), (.070, .039), (.104, .094), (.055, .088),
          (.107, .116), (.082, .100), (.148, .143), (.168, .134),
          (.245, .110), (.122, .102), (.084, .065), (.082, .081)]

C_SHARED = '0.55'
C_MG = '#0072B2'


def ratios(pairs):
    return np.array([r / n for r, n in pairs])


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
    })
    fig, ax = plt.subplots(figsize=(2.7, 2.2))
    counts = [int((ratios(d) > 1).sum())
              for d in (BASE_MEL, MG_MEL, BASE_CHD, MG_CHD)]
    x = np.array([0, 1])
    w = 0.36
    b1 = ax.bar(x - w/2, [counts[0], counts[2]], width=w*0.92,
                color=C_SHARED, edgecolor='black', linewidth=0.3,
                label='shared router')
    b2 = ax.bar(x + w/2, [counts[1], counts[3]], width=w*0.92,
                color=C_MG, edgecolor='black', linewidth=0.3,
                label='per-modality gates')
    for xi, c in zip([x[0]-w/2, x[0]+w/2, x[1]-w/2, x[1]+w/2],
                     [counts[0], counts[1], counts[2], counts[3]]):
        ax.text(xi, c + 0.25, str(c), ha='center', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Melody', 'Chord'])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 12)
    ax.set_yticks([0, 4, 8, 12])
    ax.set_ylabel('layers with content-driven\nrouting (of 12)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5)
    ax.tick_params(axis='x', length=0)
    fig.legend(frameon=False, ncol=2, loc='upper center',
               bbox_to_anchor=(0.58, 1.06), handlelength=1.1,
               handletextpad=0.45, columnspacing=1.2)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.92))
    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'content_probe.{ext}'), dpi=300,
                    bbox_inches='tight')
    print('wrote figures/content_probe.{pdf,png}')


if __name__ == '__main__':
    main()
