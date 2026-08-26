"""Paper figure: within-stream content-responsiveness of routing --
where the per-modality gates' freed capacity went.

For each layer and stream, the probe splits frames into register
terciles and measures the mean pairwise L1 between the terciles' mean
routing distributions, against a 20-permutation shuffle null (95th
pct). Plotted is the RATIO L1 / null95, so 1.0 (dashed) is the
significance threshold and the two models are comparable despite
per-condition nulls. Melody panel: both routers near ceiling -- the
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
        'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'mathtext.fontset': 'stix',
        'axes.linewidth': 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.0), sharey=True)
    layers = np.arange(12)
    w = 0.36

    for ax, title, base, mg in [
            (axes[0], 'Melody stream', BASE_MEL, MG_MEL),
            (axes[1], 'Chord stream', BASE_CHD, MG_CHD)]:
        rb, rm = ratios(base), ratios(mg)
        ax.bar(layers - w / 2, rb, width=w * 0.92, color=C_SHARED,
               edgecolor='black', linewidth=0.25, label='shared router')
        ax.bar(layers + w / 2, rm, width=w * 0.92, color=C_MG,
               edgecolor='black', linewidth=0.25,
               label='per-modality gates')
        ax.axhline(1.0, color='0.25', linewidth=0.7,
                   linestyle=(0, (4, 2)), zorder=0)
        nb, nm = int((rb > 1).sum()), int((rm > 1).sum())
        ax.set_title(f'{title}   ({nb}/12 vs {nm}/12 layers above null)',
                     pad=3)
        ax.set_xlabel('layer', labelpad=1.5)
        ax.set_xticks(layers[::2])
        ax.set_xlim(-0.7, 11.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5)

    axes[0].set_ylabel(r'register effect / null$_{95}$')
    axes[0].set_ylim(0, 2.6)
    axes[0].text(11.5, 1.03, 'shuffle null', ha='right', va='bottom',
                 fontsize=6.5, color='0.25')
    axes[1].legend(frameon=False, loc='upper left', borderpad=0.1,
                   handlelength=1.1, handletextpad=0.5, labelspacing=0.3)

    fig.tight_layout(pad=0.4, w_pad=1.2)
    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'content_probe.{ext}'), dpi=300,
                    bbox_inches='tight')
    print('wrote figures/content_probe.{pdf,png}')


if __name__ == '__main__':
    main()
