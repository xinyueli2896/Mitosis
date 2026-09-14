"""Frechet Music Distance per system, with a song-level bootstrap.

Runs INSIDE the FMD venv (frechet-music-distance + its torch), not the
mitosis env. Reads the window-cropped tree fmd_prepare.py writes,
embeds every file once with CLaMP 2, caches the embeddings, and then
does all the distance work in numpy so the bootstrap costs nothing.

Output has the same schema as eval_metrics' pooled CSV -- metric,
system, jsd, ci_lo, ci_hi, boot_se, n_songs, n_obs, n_boot, boot_mode,
weight -- so the plotter treats FMD as one more pooled metric: a single
value per system with an interval, reference level 0.

THE SAMPLE SIZE IS THE CATCH. A Frechet distance fits a Gaussian in
768 dimensions, and 93 songs x 3 samples is 279 points, so the sample
covariance is singular (rank <= 278) and the estimate is dominated by
covariance-estimation error, which falls as 1/n. Every system has the
same n, so the bias is shared and the ORDERING is more trustworthy than
the absolute value -- but whether the spread between systems clears the
noise is an empirical question. --calibrate answers it: it splits the
REFERENCE set in two and scores one half against the other, which is
what a perfect system would score at this n. Read that number first.
"""

import argparse
import csv
import os

import numpy as np
from scipy import linalg


def frechet(mu1, cov1, mu2, cov2, eps=1e-6):
    """Squared Frechet distance between two Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if not np.isfinite(covmean).all():
        # Singular product: the expected case at n < d, and what the
        # reference implementation does too.
        off = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + off).dot(cov2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2)
                 - 2 * np.trace(covmean))


def gauss(X):
    return X.mean(axis=0), np.cov(X, rowvar=False)


def embed_folder(extractor, folder, cache):
    """Embeddings for one folder, cached to .npy beside the tree."""
    if os.path.exists(cache):
        return np.load(cache)
    feats = np.asarray(extractor.extract_features(folder), dtype=np.float64)
    np.save(cache, feats)
    return feats


def song_of(fname):
    """'001__s2.mid' -> '001'; '001__w1.mid' -> '001'; '001.mid' -> '001'.

    Extra reference WINDOWS of a song share its id, so a bootstrap that
    draws that song draws all of them -- the resampling unit stays the
    song, not the excerpt.
    """
    base = os.path.splitext(fname)[0]
    return base.split('__s')[0].split('__w')[0]


def listing(folder):
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(('.mid', '.midi')))


def boot_ci(ref_X, ref_song, gen_X, gen_song, n_boot, seed=0):
    """Percentile CI, resampling SONGS -- the pooled-JSD design.

    A drawn song brings ALL its embeddings, reference and generated
    alike, so the pairing that makes the two sets comparable survives
    the resample.
    """
    songs = sorted(set(ref_song) & set(gen_song))
    if len(songs) < 4:
        return float('nan'), float('nan'), float('nan')
    ridx = {s: np.flatnonzero(ref_song == s) for s in songs}
    gidx = {s: np.flatnonzero(gen_song == s) for s in songs}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = [songs[i] for i in rng.integers(0, len(songs), len(songs))]
        r = np.concatenate([ridx[s] for s in pick])
        g = np.concatenate([gidx[s] for s in pick])
        if len(r) < 8 or len(g) < 8:
            continue
        try:
            out.append(frechet(*gauss(ref_X[r]), *gauss(gen_X[g])))
        except Exception:
            continue
    if len(out) < 2:
        return float('nan'), float('nan'), float('nan')
    v = np.asarray(out)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi), float(v.std(ddof=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tree', required=True, help='fmd_prepare.py --out')
    p.add_argument('--out', required=True, help='CSV path')
    p.add_argument('--n-boot', type=int, default=500)
    p.add_argument('--extractor', default='clamp2')
    p.add_argument('--calibrate', action='store_true', default=True)
    args = p.parse_args()

    from frechet_music_distance.models import CLaMP2Extractor
    extractor = CLaMP2Extractor(verbose=True)

    ref_dir = os.path.join(args.tree, '_reference')
    ref_files = listing(ref_dir)
    ref_X = embed_folder(extractor, ref_dir,
                         os.path.join(args.tree, '_reference.npy'))
    ref_song = np.array([song_of(f) for f in ref_files])
    print(f'[fmd] reference: {len(ref_X)} embeddings, dim {ref_X.shape[1]}')

    rows = []
    if args.calibrate:
        # What a PERFECT system scores at this n: half the reference
        # against the other half. If this is the size of the spread
        # between systems, the metric cannot separate them here and the
        # fix is more samples per song, not a different metric.
        h = len(ref_X) // 2
        a, b = ref_X[:h], ref_X[h:]
        null = frechet(*gauss(a), *gauss(b))
        print(f'[fmd] NULL CALIBRATION: reference vs itself, split {h}/'
              f'{len(ref_X)-h} = {null:.4f}')
        print('[fmd] read every system score against that floor.')
        rows.append(dict(metric='fmd', system='_null_ref_vs_ref', jsd=null,
                         ci_lo=float('nan'), ci_hi=float('nan'),
                         boot_se=float('nan'), n_songs=len(set(ref_song)),
                         n_obs=len(ref_X), n_boot=0, boot_mode='none',
                         weight='sample'))

    systems = sorted(d for d in os.listdir(args.tree)
                     if os.path.isdir(os.path.join(args.tree, d))
                     and not d.startswith('_'))
    for sysname in systems:
        folder = os.path.join(args.tree, sysname)
        files = listing(folder)
        if not files:
            print(f'[fmd] {sysname}: no files, skipped')
            continue
        X = embed_folder(extractor, folder,
                         os.path.join(args.tree, f'_{sysname}.npy'))
        song = np.array([song_of(f) for f in files])
        d = frechet(*gauss(ref_X), *gauss(X))
        lo, hi, se = boot_ci(ref_X, ref_song, X, song, args.n_boot)
        print(f'[fmd] {sysname:12s} {d:9.4f}  [{lo:.4f},{hi:.4f}]  '
              f'n={len(X)} files / {len(set(song))} songs')
        rows.append(dict(metric='fmd', system=sysname, jsd=d, ci_lo=lo,
                         ci_hi=hi, boot_se=se, n_songs=len(set(song)),
                         n_obs=len(X), n_boot=args.n_boot,
                         boot_mode='songs', weight='sample'))

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'system', 'jsd', 'ci_lo',
                                          'ci_hi', 'boot_se', 'n_songs',
                                          'n_obs', 'n_boot', 'boot_mode',
                                          'weight'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
