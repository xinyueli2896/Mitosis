"""Paper figure: per-expert top-1 routing load by layer, for the three
A.2 router designs -- the "are there dead experts?" figure.

Every bar rises from zero, so a dead expert (one that never wins any
token's top-1) would be a visibly missing bar. Across all three designs
and all 12 layers, no (layer, expert) pair is at zero, and loads sit
near the balanced ideal 1/E = 0.25 (dashed line).

Data: per-expert top-1 load tables from analyze_moe_routing.sbatch --
  shared        job 182681 (seeded batch, SEED=0)
  + bias (mb)   job 179683
  per-modality gates (mg)  job 182680 (seeded batch, SEED=0)
The utilization claim replicates across every batch analyzed (jobs
178410, 178528, 178945/46, 179683/84, 182634/35, 182665/66, 182680/81):
zero dead experts in all of them.

Run anywhere with matplotlib (no torch needed):
    python figures/plot_expert_load.py
Writes figures/expert_load.pdf (vector, for LaTeX) and .png (300 dpi).
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, colorblind-safe; fixed expert order e0..e3.
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']

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


def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), sharey=True)
    layers = np.arange(12)
    width = 0.19

    for ax, (title, data) in zip(axes, PANELS):
        arr = np.array(data)
        for e in range(4):
            ax.bar(layers + (e - 1.5) * width, arr[:, e], width=width * 0.92,
                   color=COLORS[e], edgecolor='black', linewidth=0.25,
                   label=f'expert {e}')
        ax.axhline(0.25, color='0.25', linewidth=0.7, linestyle=(0, (4, 2)),
                   zorder=0)
        ax.set_title(title, pad=3)
        ax.set_xlabel('layer', labelpad=1.5)
        ax.set_xticks(layers[::2])
        ax.set_xlim(-0.7, 11.7)
        ax.set_ylim(0, 0.5)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5)

    axes[0].set_ylabel('top-1 load')
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='0.25', linewidth=0.7,
                          linestyle=(0, (4, 2))))
    labels.append(r'balanced $1/E$')
    fig.legend(handles, labels, ncol=5, frameon=False,
               loc='upper center', bbox_to_anchor=(0.5, 1.09),
               handlelength=1.1, handletextpad=0.45, columnspacing=1.4)

    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.95))
    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'expert_load.{ext}'), dpi=300,
                    bbox_inches='tight')
    print('wrote figures/expert_load.{pdf,png}')


if __name__ == '__main__':
    main()
