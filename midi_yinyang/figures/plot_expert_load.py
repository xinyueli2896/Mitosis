"""Paper figure: expert utilization -- the "are there dead experts?"
figure, as a distribution strip.

One dot per (layer, expert) pair: 12 layers x 4 experts = 48 dots per
router design. The claim reads directly off the geometry: no dot sits
at zero (a dead expert would), every cloud hugs the balanced ideal
1/E = 0.25, and the per-modality-gates cloud is the tightest. The
per-design minimum is annotated -- the single number the dead-expert
question reduces to.

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

DESIGNS = [('Shared', SHARED), ('+ Bias', MB), ('Per-modality\ngates', MG)]
DOT = '#0072B2'


def paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })


def main():
    paper_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    rng = np.random.default_rng(0)

    for i, (name, data) in enumerate(DESIGNS):
        vals = np.array(data).ravel()
        x = i + rng.uniform(-0.16, 0.16, size=vals.size)
        ax.scatter(x, vals, s=9, facecolor=DOT, edgecolor='none',
                   alpha=0.65, zorder=3)
        med, lo = np.median(vals), vals.min()
        ax.hlines(med, i - 0.24, i + 0.24, color='black', linewidth=1.1,
                  zorder=4)
        ax.annotate(f'min {lo:.2f}', xy=(i, lo), xytext=(i + 0.02, lo - 0.045),
                    ha='center', fontsize=6.5, color='0.25')

    ax.axhline(0.25, color='0.25', linewidth=0.7, linestyle=(0, (4, 2)),
               zorder=1)
    ax.text(2.44, 0.256, r'balanced $1/E$', ha='right', va='bottom',
            fontsize=6.5, color='0.25')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.text(2.44, 0.008, 'dead-expert level', ha='right', va='bottom',
            fontsize=6.5, color='0.25')

    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, _ in DESIGNS])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.01, 0.47)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylabel('per-expert top-1 load')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5)
    ax.tick_params(axis='x', length=0)

    fig.tight_layout(pad=0.4)
    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'expert_load.{ext}'), dpi=300,
                    bbox_inches='tight')
    print('wrote figures/expert_load.{pdf,png}')


if __name__ == '__main__':
    main()
