"""Paper figures for the C.1 rehearsal reconstruction study.

Figure 1 (rehearsal_dissociation): the two claims that carry the study,
on the chord->melody direction where all three variants exist.
  (a) Teacher-forced vs free-running reconstruction of the CONDITIONING
      stream. Teacher-forced scores are ~1.0 for the legacy geometry and
      for C.1.A alike -- indistinguishable. Free-running separates them:
      the legacy model collapses to 0.830 while C.1.A holds 0.999. With
      clean context a shift-2 model can CONTINUE a sustained chord
      instead of retrieving it from the prefix; only free-running, where
      its own errors must be re-anchored, exposes the broken addressing.
  (b) The two design axes decouple. x = free-running reconstruction of
      the conditioning stream (governed by the prefix geometry),
      y = teacher-forced prediction of the generation stream (governed
      by the suffix shift). Legacy generates well but cannot retrieve;
      C.1.B retrieves but its generation collapses; C.1.A is the only
      configuration strong on both axes.

Figure 2 (rehearsal_direction): the same two metrics for C.1.A / C.1.B
in BOTH directions, so the mechanism claim is separated from content
difficulty. Retrieval holds in both directions for both variants. The
generation gap is direction-dependent: C.1.B's shift-1 collapses melody
generation (0.647) but not chord generation (0.993) -- chords are dense
and largely predictable from the conditioning melody, melody is not. The
shift penalty therefore lands on the HARD generation task.

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

C_LEGACY = '0.55'
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
        'axes.titlesize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'mathtext.fontset': 'stix',
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })


def fig_dissociation(out):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5))

    # (a) teacher-forced vs free-running, conditioning-stream recon
    ax = axes[0]
    names = list(REV)
    for i, n in enumerate(names):
        d = REV[n]
        ax.plot([i, i], [d['tf'], d['free']], color=d['color'],
                linewidth=1.0, zorder=2)
        ax.scatter([i], [d['tf']], s=26, marker='o', facecolor='white',
                   edgecolor=d['color'], linewidth=1.3, zorder=3)
        ax.scatter([i], [d['free']], s=26, marker='o', facecolor=d['color'],
                   edgecolor='black', linewidth=0.3, zorder=3)
        ax.annotate(f'{d["free"]:.3f}', xy=(i, d['free']),
                    xytext=(i + 0.17, d['free']), fontsize=6.5,
                    va='center', color='0.25')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlim(-0.45, len(names) - 0.25)
    ax.set_ylim(0.78, 1.03)
    ax.set_ylabel('conditioning-stream reconstruction')
    ax.set_title('(a) teacher-forced hides the failure', pad=4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5)
    ax.tick_params(axis='x', length=0)
    h_tf = plt.Line2D([], [], marker='o', linestyle='none', markersize=5,
                      markerfacecolor='white', markeredgecolor='0.35',
                      label='teacher-forced')
    h_fr = plt.Line2D([], [], marker='o', linestyle='none', markersize=5,
                      markerfacecolor='0.35', markeredgecolor='black',
                      markeredgewidth=0.3, label='free-running')
    ax.legend(handles=[h_tf, h_fr], frameon=False, loc='lower left',
              borderpad=0.1, handletextpad=0.4, labelspacing=0.25)

    # (b) the two axes decouple
    ax = axes[1]
    for n, d in REV.items():
        ax.scatter([d['free']], [d['gen']], s=46, facecolor=d['color'],
                   edgecolor='black', linewidth=0.4, zorder=3)
    ax.annotate('legacy\n(geometry bug)', xy=(REV['legacy']['free'],
                REV['legacy']['gen']), xytext=(0.845, 0.905),
                fontsize=7, ha='left', color='0.25')
    ax.annotate('C.1.A', xy=(REV['C.1.A']['free'], REV['C.1.A']['gen']),
                xytext=(0.994, 0.918), fontsize=7.5, ha='right',
                fontweight='bold', color=C_A)
    ax.annotate('C.1.B\n(shift-1)', xy=(REV['C.1.B']['free'],
                REV['C.1.B']['gen']), xytext=(0.862, 0.672),
                fontsize=7, ha='left', color='0.25')
    ax.set_xlabel('conditioning recon (free-running)')
    ax.set_ylabel('generation accuracy (teacher-forced)')
    ax.set_title('(b) geometry and shift govern different axes', pad=4)
    ax.set_xlim(0.79, 1.04)
    ax.set_ylim(0.60, 1.02)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5)
    ax.annotate('', xy=(1.005, 1.005), xytext=(0.930, 0.982),
                arrowprops=dict(arrowstyle='->', lw=0.7, color='0.5'))
    ax.text(0.923, 0.982, 'ideal', fontsize=6.5, color='0.5',
            va='center', ha='right')

    fig.tight_layout(pad=0.4, w_pad=1.6)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'rehearsal_dissociation.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('wrote figures/rehearsal_dissociation.{pdf,png}')


def fig_direction(out):
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), sharex=True)
    x = np.array([0, 1])
    w = 0.32
    dirs = ['mel' + r'$\rightarrow$' + 'chord', 'chord' + r'$\rightarrow$' + 'mel']

    for ax, key, title, ylim in (
            (axes[0], 'free', '(a) conditioning reconstruction\n(free-running)',
             (0.80, 1.03)),
            (axes[1], 'gen', '(b) generation accuracy\n(teacher-forced)',
             (0.60, 1.05))):
        a = [FWD['C.1.A'][key], REV['C.1.A'][key]]
        b = [FWD['C.1.B'][key], REV['C.1.B'][key]]
        # Dots, not bars: the informative range is 0.6-1.0 and a
        # truncated bar axis would misrepresent length-from-zero.
        ax.scatter(x - w / 2, a, s=40, marker='o', facecolor=C_A,
                   edgecolor='black', linewidth=0.35, zorder=3,
                   label='C.1.A (shift-2)')
        ax.scatter(x + w / 2, b, s=40, marker='s', facecolor=C_B,
                   edgecolor='black', linewidth=0.35, zorder=3,
                   label='C.1.B (shift-1)')
        for xi, v in zip(x - w / 2, a):
            ax.text(xi, v + 0.017, f'{v:.3f}', ha='center', fontsize=6.3,
                    color='0.25')
        for xi, v in zip(x + w / 2, b):
            ax.text(xi, v + 0.017, f'{v:.3f}', ha='center', fontsize=6.3,
                    color='0.25')
        for xi in x:                       # light guide per direction
            ax.axvline(xi, color='0.9', linewidth=0.6, zorder=0)
        ax.set_title(title, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(dirs)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(*ylim)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5)
        ax.tick_params(axis='x', length=0)

    axes[0].set_ylabel('accuracy / Jaccard')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc='upper center',
               bbox_to_anchor=(0.55, 1.10), handlelength=1.0,
               handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0, 1, 0.93))
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
