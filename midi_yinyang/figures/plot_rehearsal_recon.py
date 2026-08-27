"""Paper figures for the C.1 rehearsal reconstruction study.

Both figures are grouped bar charts from a true zero baseline, with one
consistent color encoding: gray = legacy geometry, blue = C.1.A
(shift-2), orange = C.1.B (shift-1). Every bar carries its value, so the
truncation-free axis costs no precision.

Figure 1 (rehearsal_dissociation): chord->melody, where all three
variants exist -- one group per metric. Reading left to right is the
study's argument: teacher-forced reconstruction cannot tell the variants
apart (~1.0 everywhere); free-running separates them (legacy collapses
to 0.830, C.1.A holds 0.999); generation accuracy is governed by the
OTHER axis (C.1.B's shift-1 collapses it to 0.647). Geometry moves only
the middle group, shift only the right group: the axes decouple.

Figure 2 (rehearsal_direction): C.1.A / C.1.B in BOTH directions, so the
mechanism claim is separated from content difficulty. Panel (a):
retrieval holds everywhere except C.1.B on dense chords (0.847), the
pure-retrieval instrument meeting the hard content. Panel (b): C.1.B's
generation deficit is direction-dependent -- 0.647 generating melody,
0.993 generating chords. The shift penalty lands on the HARD task.

Data (VARIANTS.md, "C.1 reconstruction study"): rev = jobs 178861-178864
and 178877-178878; fwd = jobs 179919/179920 (teacher-forced) and
179956/179957 (free-running). Generation accuracy is the analyzer's
generative-floor token accuracy (fwd values from the printed floors:
1 - 0.0201 and 1 - 0.0066).

Run: python figures/plot_rehearsal_recon.py   (matplotlib only)
Writes figures/rehearsal_{dissociation,direction}.{pdf,png}
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

C_LEGACY = '0.62'
C_A = '#0072B2'
C_B = '#E69F00'

# chord -> melody (conditioning = chord, generation = melody)
REV = {
    'legacy':  dict(tf=0.998, free=0.830, gen=0.961, color=C_LEGACY),
    'C.1.A':   dict(tf=1.000, free=0.999, gen=0.952, color=C_A),
    'C.1.B':   dict(tf=0.950, free=0.847, gen=0.647, color=C_B),
}
# melody -> chord (conditioning = melody, generation = chord)
FWD = {
    'C.1.A':   dict(tf=0.998, free=1.000, gen=0.980, color=C_A),
    'C.1.B':   dict(tf=0.999, free=0.997, gen=0.993, color=C_B),
}


def paper_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 7, 'legend.fontsize': 7,
        'mathtext.fontset': 'stix', 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    })


def style_value_axis(ax):
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '.25', '.5', '.75', '1'])
    ax.grid(axis='y', color='0.88', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5)
    ax.tick_params(axis='x', length=0)


def bar_label(ax, x, v):
    ax.text(x, v + 0.022, f'{v:.3f}'.lstrip('0'), ha='center',
            fontsize=6.3, color='0.2')


def fig_dissociation(out):
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    metrics = [('tf', 'cond. reconstruction\n(teacher-forced)'),
               ('free', 'cond. reconstruction\n(free-running)'),
               ('gen', 'generation accuracy\n(teacher-forced)')]
    variants = list(REV)
    x = np.arange(len(metrics))
    w = 0.26
    for j, name in enumerate(variants):
        d = REV[name]
        xs = x + (j - 1) * (w + 0.015)
        vals = [d[k] for k, _ in metrics]
        label = {'legacy': 'legacy (geometry bug)',
                 'C.1.A': 'C.1.A (shift-2)',
                 'C.1.B': 'C.1.B (shift-1)'}[name]
        ax.bar(xs, vals, width=w, color=d['color'], edgecolor='black',
               linewidth=0.3, label=label, zorder=2)
        for xi, v in zip(xs, vals):
            bar_label(ax, xi, v)
    style_value_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    ax.set_ylabel('accuracy / Jaccard')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc='upper center',
               bbox_to_anchor=(0.55, 1.06), handlelength=1.0,
               handletextpad=0.4, columnspacing=1.3)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'rehearsal_dissociation.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('wrote figures/rehearsal_dissociation.{pdf,png}')


def fig_direction(out):
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2), sharey=True)
    dirs = ['mel' + r'$\rightarrow$' + 'chord',
            'chord' + r'$\rightarrow$' + 'mel']
    x = np.arange(2)
    w = 0.30
    for ax, key, title in (
            (axes[0], 'free', '(a) cond. reconstruction (free-running)'),
            (axes[1], 'gen', '(b) generation accuracy (teacher-forced)')):
        a = [FWD['C.1.A'][key], REV['C.1.A'][key]]
        b = [FWD['C.1.B'][key], REV['C.1.B'][key]]
        ax.bar(x - (w + 0.02) / 2, a, width=w, color=C_A,
               edgecolor='black', linewidth=0.3, zorder=2,
               label='C.1.A (shift-2)')
        ax.bar(x + (w + 0.02) / 2, b, width=w, color=C_B,
               edgecolor='black', linewidth=0.3, zorder=2,
               label='C.1.B (shift-1)')
        for xi, v in zip(x - (w + 0.02) / 2, a):
            bar_label(ax, xi, v)
        for xi, v in zip(x + (w + 0.02) / 2, b):
            bar_label(ax, xi, v)
        style_value_axis(ax)
        ax.set_title(title, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(dirs)
        ax.set_xlim(-0.55, 1.55)
    axes[0].set_ylabel('accuracy / Jaccard')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc='upper center',
               bbox_to_anchor=(0.55, 1.08), handlelength=1.0,
               handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0, 1, 0.94))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'rehearsal_direction.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('wrote figures/rehearsal_direction.{pdf,png}')


def main():
    paper_style()
    out = os.path.dirname(os.path.abspath(__file__))
    fig_dissociation(out)
    fig_direction(out)


if __name__ == '__main__':
    main()
