# Paper notes — abstract versions and what was cut for length

Terminology (settled): melody and chords are *streams* (general term,
`\strm{}` / `\strms{}` macros); *modality* appears only in the
positioning passage; *domain* = corpus; *part* is music-only prose.

## Short abstract (2026-09-09, ~200 words)

```latex
% \newcommand{\strm}{\textit{stream}}  \newcommand{\strms}{\textit{streams}}
\begin{abstract}
Cross-modal generation is posed at two distances: far, bridged by
learned alignment (vision and language); near, sharing content but not
form (text and speech). Generating both signals jointly, rather than
one given the other, raises a question conditioning never faces: each
signal follows its own grammar and needs its own computation, yet the
two must agree at every instant. Joint models settle it in advance,
sharing everything or partitioning parameters by modality, and cannot
test the choice: paired data is scarce, so no fully shared model on
the same data exists as a yardstick. We therefore study two \strms{}
of one modality, melody and chords in symbolic music: one
representation and one clock, two grammars, and abundant paired data
on which the fully shared generator is the established design. We
propose a duet transformer, one autoregressive model over the
interleaved \strms{}, \strm{}-specific in its attention projections
and its router over a shared, unassigned expert pool, so
specialisation is learned, not imposed; appended query slots let each
\strm{} condition on its partner's current frame. On held-out songs no
compared system trained on, one checkpoint generates both \strms{} in
either direction, matching the fully shared generator and cascades in
quality while keeping the \strms{} coupled.
\end{abstract}
```

The results sentence is provisional until the merged 8-song table
with the decode diagnostics and the cp8 retrains is in (EXPERIMENTS.md,
"E1 follow-up").

## Cut from the long abstract — expand these in the introduction

1. **The conditional norm.** "At either distance the norm is to
   generate one signal given the other in full, which fixes the
   direction and forgoes coupling at each step." Introduction §1:
   captioning, text-to-image, TTS, ASR as the conditional line; what
   conditioning gives up (direction fixed at the split, the given
   signal never written with the partner in view).
2. **Joint-generation examples.** "as recent speech--text and
   audio--visual models do" — cite Moshi (speech + text, one clock) and
   an audio-visual joint generator; these are the synchronous pairs for
   which "agree at every instant" is literal.
3. **How joint models settle the sharing question.** "sharing all
   parameters and marking the signals with an identity token, or
   partitioning parameters by modality; where a signal's computation
   should diverge is imposed, never learned." Introduction §2: the
   shared pole (Chameleon, Moshi; Transfusion with modality-specific
   boundary layers and attention pattern) vs the partitioned pole
   (Mixture-of-Transformers, LMFusion, BAGEL) and imposed experts
   (MoMa). None learns the split; none is compared against its own
   alternative on the same data.
4. **Why the yardstick is missing across modalities.** "models build on
   pretrained unimodal models and learn only the bridge" — one paragraph:
   paired data scarce relative to unimodal, so every cross-modal design
   starts from unimodal pretraining and only the bridge is trained on
   pairs; any comparison is confounded with pretraining.
5. **Stream vs modality, in full.** "which share one representation,
   its vocabulary and clock, yet obey their own grammars, as distinct in
   content as two modalities." The passive-aggressive defence: melody
   and chords have the property that makes two modalities interesting
   (own content, own grammar) and lack the one that gets in the way
   (different representation). Also the transfer claim, cut earlier:
   cross-modal pairs reach the same condition once tokenised into one
   vocabulary and interleaved (Moshi, AudioPaLM), so the answer applies
   to them past the point of alignment.
6. **Second reason the testbed is clean** (cut two rounds ago): because
   the streams share one representation, any specialisation the routers
   learn is by grammar, not by input statistics; in a vision–language
   mixture modality-specific experts separate on pixels-vs-words alone.
7. **Query-slot mechanism detail.** "masked predictors of the current
   frame refined iteratively at inference" — method section: slots
   trained under random masking of either stream's current frame
   (commitment levels), decoded by iterative refinement or
   commit-then-condition.
8. **Yardstick wording.** "which provides the yardstick" folded into
   "the fully shared generator is the established design"; the
   introduction should state explicitly that the fully shared model is
   the null hypothesis the paper has to beat or explain.

## Long abstract (2026-09-08, ~280 words), for reference

Generation across modalities is posed at two distances: far, bridged
by learned alignment (vision and language); near, sharing content but
not form (text and speech). At either distance the norm is to generate
one signal given the other in full, which fixes the direction and
forgoes coupling at each step. Generating both jointly, as recent
speech--text and audio--visual models do, raises a question
conditioning never faces: each signal follows its own grammar and
needs its own computation, yet the two must still agree at every
instant. Joint models settle it in advance, sharing all parameters and
marking the signals with an identity token, or partitioning parameters
by modality; where a signal's computation should diverge is imposed,
never learned. Whether signal-specific computation helps is hard to
test across modalities: paired data is scarce, so models build on
pretrained unimodal models and learn only the bridge; no fully shared
model on the same data exists as a yardstick. We therefore study two
streams of one modality, melody and chords in symbolic music, which
share one representation, its vocabulary and clock, yet obey their own
grammars, as distinct in content as two modalities. Paired data is
abundant here, and the fully shared generator trained on it is the
established design, which provides the yardstick. We propose a duet
transformer: one autoregressive model over the two interleaved
streams, stream-specific in its attention projections and its router
over a shared, unassigned expert pool, so specialisation by stream is
learned, not imposed. Appended query slots, masked predictors of the
current frame refined iteratively at inference, let each stream
condition on its partner's current frame. On held-out songs no
compared system trained on, one checkpoint generates both streams in
either direction, matching the fully shared generator and cascades in
quality while keeping the streams coupled.
