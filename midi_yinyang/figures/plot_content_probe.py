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
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.7), sharey=True)
    layers = np.arange(12)

    for ax, title, base, mg in [
            (axes[0], 'Melody stream', BASE_MEL, MG_MEL),
            (axes[1], 'Chord stream', BASE_CHD, MG_CHD)]:
        rb, rm = ratios(base), ratios(mg)
        for l in layers:
            ax.plot([rb[l], rm[l]], [l, l], color='0.75', linewidth=0.9,
                    zorder=1)
        ax.scatter(rb, layers, s=16, facecolor=C_SHARED, edgecolor='black',
                   linewidth=0.3, zorder=3, label='shared router')
        ax.scatter(rm, layers, s=16, marker='s', facecolor=C_MG,
                   edgecolor='black', linewidth=0.3, zorder=3,
                   label='per-modality gates')
        ax.axvline(1.0, color='0.25', linewidth=0.7,
                   linestyle=(0, (4, 2)), zorder=0)
        nb, nm = int((rb > 1).sum()), int((rm > 1).sum())
        ax.set_title(f'{title}\n{nb}/12 vs {nm}/12 layers above null',
                     pad=3)
        ax.set_xlabel(r'register effect / null$_{95}$', labelpad=1.5)
        ax.set_xlim(0, 2.7)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5)

    axes[0].set_ylim(11.6, -0.6)                 # layer 0 on top
    axes[0].set_yticks(layers)
    axes[0].set_yticklabels([str(l) for l in layers])
    axes[0].set_ylabel('layer')
    from matplotlib.lines import Line2D
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='0.25', linewidth=0.7,
                          linestyle=(0, (4, 2))))
    labels.append('shuffle null')
    fig.legend(handles, labels, frameon=False, ncol=3,
               loc='lower center', bbox_to_anchor=(0.54, -0.04),
               handlelength=1.2, handletextpad=0.45, columnspacing=1.6)

    fig.tight_layout(pad=0.4, w_pad=1.0)
    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'content_probe.{ext}'), dpi=300,
                    bbox_inches='tight')
    print('wrote figures/content_probe.{pdf,png}')


if __name__ == '__main__':
    main()
