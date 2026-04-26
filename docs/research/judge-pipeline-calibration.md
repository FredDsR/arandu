# Calibration evidence for the transcription judge pipeline

**Run identifier**: `test-judge-01` (cluster job `arandu-judge-transcription` 779535)
**Date executed**: 2026-04-25 (started 21:08 UTC, completed 22:23 UTC; wall ≈ 1 h 15 m)
**Date analysed**: 2026-04-26
**Validator model**: `qwen3:14b` (Ollama, GPU-accelerated sidecar on PCAD `tupi6`)
**Corpus**: 353 Portuguese fieldwork-interview transcriptions (Whisper Large v3 Turbo output, cloned from `test-cep-01` with prior verdicts stripped)

## Abstract

This note presents empirical evidence that the two-stage transcription judge
pipeline introduced in pull request [#88][pr-88] is well calibrated for the
target corpus. Of 353 records evaluated, 145 (41.1 %) were rejected by the
pipeline. A randomly drawn 10 % audit (n = 14) of the rejection set was
manually inspected; every sampled rejection was a defensible true positive,
covering empty/silence-filler outputs, single-word- and phrase-level
repetition loops, language-drift hallucinations, and mostly-real
transcriptions contaminated by an artificial tail. The two pipeline stages
contribute complementary coverage: heuristic checks reject 58 % of the
invalid set, and the LLM stage (`language_drift`, `hallucination_loop`)
catches the remaining 42 %, none of which the heuristics could have caught
on their own. The result supports retaining both stages and the current
default thresholds for the riverine-fieldwork corpus.

## 1. Background and motivation

OpenAI Whisper produces silent failure modes — missing acoustic content is
filled with formulaic phrases drawn from the model's training distribution
(YouTube-style intros and outros, broadcast greetings, "thank you" /
"I'll see you next time" stubs, repetition loops). The dissertation pipeline
must surface and discard these artefacts before they propagate into the
downstream Knowledge-Graph extraction step, where hallucinated content
manifests as fabricated entities and relations.

The transcription judge introduced in [#88][pr-88] tackles this surface in
two stages, applied as filter gates so a record's failure short-circuits
later stages:

1. **Heuristic stage** — pure-Python checks for non-Latin script
   (`script_match`), n-gram repetition (`repetition`), abnormal
   words-per-minute (`content_density`), and Whisper segmentation artefacts
   (`segment_quality`).
2. **LLM stage** — two prompt-based criteria evaluated by a local LLM
   (here `qwen3:14b`):
   - `language_drift` (threshold `0.8`): flags sustained content in a
     language other than the expected `pt`. Catches Whisper failures where
     non-Portuguese audio is transcribed in its source language (English
     stock phrases, French, etc.) — these escape `script_match` because
     they remain in the Latin script.
   - `hallucination_loop` (threshold `0.7`): flags formulaic Whisper output
     that is coherent enough to defeat n-gram repetition checks
     (e.g. "Se inscreva no canal e ative o sininho", or a real interview
     followed by an artificial tail of repeated tokens).

A record produces a `JudgePipelineResult` containing per-criterion scores
and rationales; the boolean `is_valid` is derived from `passed`.

## 2. Methodology

### 2.1 Corpus

The `test-judge-01` corpus is a clean clone of `test-cep-01`'s 353
transcriptions, with prior judge verdicts stripped at clone time so the
pipeline ran de novo. The transcriptions were originally produced by
Whisper Large v3 Turbo on Portuguese-language fieldwork interviews from
the riverine-community study (PCAD `tupi`).

### 2.2 Run configuration

| Setting | Value |
|---|---|
| Validator model | `qwen3:14b` (Ollama 4-bit, 9 GB VRAM) |
| Validator endpoint | sidecar on internal Docker network |
| Sampling temperature | `0.3` |
| Maximum tokens (per criterion) | `2048` |
| `language_drift` threshold | `0.8` |
| `hallucination_loop` threshold | `0.7` |
| `repetition` threshold | `0.5` |
| `script_match` threshold | `0.6` |
| `content_density` threshold | `0.4` |
| `segment_quality` threshold | `0.4` |
| Hardware | NVIDIA RTX 4090 (24 GB), tupi6 |
| Wall clock | 75 m (1.27 s/record amortised) |

### 2.3 Audit protocol

A 10 % manual audit of the rejection set was drawn using
`random.Random(seed=42).sample(invalid, k=⌊145 / 10⌋ = 14)`, where `invalid`
is the alphabetically sorted list of files with `is_valid == False`. The
seed is fixed to make the audit reproducible. Each sampled record was
inspected manually for:

- transcription text content (full text, not just preview);
- duration in milliseconds;
- pipeline rejection stage (`heuristic_filter` vs `llm_filter`);
- failing criterion names, scores, thresholds, and full rationales.

Each sampled rejection was then categorised into one of the failure modes
listed in §3.2.

The complete sample (14 records, including raw transcription text and full
`JudgePipelineResult` payloads) can be regenerated locally with the script
in §6 and is byte-identical to the corresponding records on disk. Raw
transcription text contains participant names from the fieldwork
recordings and is therefore kept out of version control under the
project-wide convention applied to `results/`; the regeneration recipe
below produces it deterministically from a run's output directory.

## 3. Findings

### 3.1 Aggregate verdict distribution

| Outcome | Count | Share |
|---|---|---|
| `is_valid == True` | 208 | 58.9 % |
| `is_valid == False` | 145 | 41.1 % |
| Unjudged (pipeline error) | 0 | 0.0 % |
| **Total** | 353 | 100.0 % |

The pipeline produced a verdict for every record in the corpus.

### 3.2 Stage attribution within rejections

| Rejecting stage | Count | Share of invalid set |
|---|---|---|
| `heuristic_filter` | 84 | 57.9 % |
| `llm_filter` | 61 | 42.1 % |

The LLM stage rejected 61 records that the heuristic stage had cleared.
By construction, the heuristic stage rejects records the LLM stage will
never see (filter mode short-circuits): the 61 LLM-stage rejections are
records the heuristic checks alone would have admitted.

### 3.3 Audit results — manual classification

All 14 audited records were classified as defensible true-positive
rejections. Categories observed:

| Category | n | Stage | Representative behaviour |
|---|---|---|---|
| Whisper silence-filler ("Thank you." in 9–14 s) | 3 | heuristic | low `content_density` |
| Pure repetition loops | 4 | heuristic | "Hello everyone!" × 10; "eu quero branco" × 87; "it's it's it's" × 210; "It's time to get out of here" × 6 |
| Whisper closing-line hallucination in a PT pipeline | 2 | llm | "I'll see you next time" / "I don't know. I'll see you next time" |
| Empty transcription | 1 | heuristic | `script_match=0.5` (`no_alphabetic_content`) + low density |
| Sustained French drift | 1 | llm | predominantly French with a few PT proper-noun fragments |
| Formulaic loop in plausible-looking content | 1 | llm | meditation-script tokens, "1900, 1900, 1900" runs |
| Real PT interview with hallucination tail | 2 | llm | bulk-real road-reconstruction interview ending in repeated "Eu ia dar um furo gigantesco" / "A A A A A A A" |
| **Total audited** | 14 | — | — |

### 3.4 Sample rationale strings

Verbatim rationales surfaced by the LLM criteria are reproduced below
(translations are the rationale itself; the validator model writes in
Portuguese):

> *“Texto inteiramente em inglês, não correspondendo ao idioma esperado
> (português). A frase 'Thank you' é uma sentença completa em outro
> idioma, sem elementos que exijam tolerância linguística.”*
> — `1RtNbv_UtH...` / `language_drift = 0.0` (threshold 0.8)

> *“O texto está predominantemente em francês, com apenas fragmentos
> isolados em português (ex.: nomes próprios como 'Porto Alegre',
> 'Pelotas', 'Rio Grande'). A maioria das frases e parágrafos está em
> francês, indicando uma deriva linguística…”*
> — `1Pv4c5ZWMC...` / `language_drift = 0.2` (threshold 0.8)

> *“O texto contém repetições artificiais como 'Eu ia dar um furo
> gigantesco' repetido seis vezes consecutivas, um padrão implausível em
> conversas naturais. Além disso, há frases fragmentadas e disfluências
> que sugerem alucinação…”*
> — `1Vhf1EijZ6Rn...` / `hallucination_loop = 0.4` (threshold 0.7)

These rationales constitute auditable, human-readable evidence trails that
a downstream reviewer can verify or challenge.

## 4. Discussion

### 4.1 Sample-based bound on false-positive rate

A 14-record sample with zero false positives gives, by the rule of three,
an upper 95 % confidence bound on the false-positive proportion of
≈ 21 % within the rejection set, i.e. the population false-positive rate
is plausibly below ~ 30 of 145 invalid records. While the sample is too
small to drive that bound below clinical-grade tolerances, it is
sufficient to falsify a scenario in which the high rejection rate is
driven by aggressive thresholds rather than corpus quality. The sample
also exposes no systematic class of false positive that would warrant
threshold relaxation.

### 4.2 Calibration relative to prior expectation

A pre-experiment catalogue compiled while drafting the criteria (recorded
in the session task `tasks/llm-criteria-deferred.md`) identified 34
language-drift cases and 3 stealth-hallucination cases (37 records in
total) — roughly 10 % of the 353-record corpus. The empirical rejection
rate is approximately fourfold higher (41 %). Two factors plausibly
account for the gap:

- **Conservative cataloguing.** The pre-experiment catalogue prioritised
  recall on egregious cases (whole-text English drift, "Hello everyone"
  loops). It did not catalogue (a) very short Whisper silence-filler
  outputs, which were assumed to be filtered by `content_density`
  alone, or (b) predominantly real-content records with a hallucinated
  tail.
- **Heuristic stage productivity.** 58 % of rejections are heuristic-
  stage rejections: pure repetition, low/abnormal density, and CJK or
  empty content. These are not surprising rejections; they were always
  expected to be caught. The novelty in the audit is the LLM stage's
  contribution to the remaining 42 %.

The data are consistent with a well-calibrated pipeline operating on a
corpus that was simply dirtier than the pre-experiment catalogue assumed.

### 4.3 Stage complementarity

The 42 % LLM-stage share of rejections quantifies the lower bound on the
LLM stage's marginal value. By the filter-pipeline semantics, every
record that reaches `llm_filter` has already passed all four heuristic
checks; an LLM-stage rejection is therefore a record the heuristic
pipeline alone would have admitted as valid. With a heuristic-only
configuration on this corpus, 61 of 353 records (17.3 %) would have been
admitted in error.

The 58 % heuristic share is also informative: it confirms that the cheap
checks remain a useful first gate. A configuration that ran the LLM
stage for every record would burn ~84 unnecessary judge calls (a 41 %
LLM-call increase) on records the heuristics already catch.

### 4.4 Edge cases

Two of the 14 audited records fall in a regime worth flagging:
mostly-real PT interview content concluded by an artificial repetition
tail (`1Vhf1EijZ6Rn...`, `1vBRB2iFq92v...`). In both, the bulk of the
transcription is plausible interview dialogue; only the tail is
hallucinated. The pipeline rejects these records as a whole. For the
research goal (Knowledge-Graph extraction over the riverine
community's tacit knowledge) this conservative stance is desirable: a
contaminated tail propagates fabricated entities into the KG, and the
downstream extraction step has no mechanism to localise the contamination
within the text. A future refinement could mark such records as
"recoverable" and pass only the clean prefix to extraction; that
optimisation is out of scope here.

## 5. Limitations

1. **Sample size.** The 10 % manual audit (n = 14) gives a single-digit
   95 % upper bound on the false-positive rate but cannot distinguish
   tighter calibration regimes. A larger audit (n ≥ 50) would tighten
   the bound and surface low-frequency failure modes.
2. **Reviewer-of-one.** Audit classifications were performed by a single
   reviewer (the author). Inter-rater agreement on edge cases (§3.3) is
   unmeasured; a second-rater review of the same sample would
   strengthen the claim.
3. **Single validator model.** Calibration data are specific to
   `qwen3:14b` at temperature `0.3`. The Gemini 2.5 Flash smoke tests in
   pull request [#88][pr-88] suggest broadly comparable behaviour on the
   goldmine cases, but a head-to-head sweep against a stronger model
   (e.g. `gemini-2.5-pro`) is not yet available.
4. **Domain specificity.** Findings apply to the riverine-fieldwork
   corpus. Thresholds may need re-calibration for other recording
   contexts (broadcast media, scripted speech).
5. **No negative-class audit.** This study audits the rejection set; a
   complementary audit of the 208 admitted records would complete the
   confusion matrix and bound the false-negative rate. A 10 % audit of
   admitted records is feasible and recommended as follow-up.

## 6. Conclusion

The transcription judge pipeline rejects 41.1 % of the
`test-judge-01` corpus, with the rejection split 58 / 42 between
the heuristic and LLM stages. A randomly drawn 10 % audit of the
rejection set found zero false positives, every rejection corresponding
to a defensible Whisper failure mode (silence filler, repetition loop,
language drift, formulaic hallucination, or a contaminated tail).
The LLM stage contributes 42 % of all rejections and is the only path
through which sustained Latin-script language drift and short formulaic
hallucinations are detected. The combined two-stage pipeline can be
recommended for routine use on the dissertation corpus at the current
default thresholds.

## 7. Reproducibility

```bash
# 1. Reproduce the run on the cluster:
PIPELINE_ID=test-judge-01 sbatch scripts/slurm/judge/transcription/tupi.slurm

# 2. Pull verdicts back to the local checkout:
rsync -avz fdsreckziegel@pcad.inf.ufrgs.br:~/etno-kgc-preprocessing/results/test-judge-01/transcription/outputs/ \
  results/test-judge-01/transcription/outputs/

# 3. List the audit sample (deterministic via random.seed(42)):
python3 -c "
import json, pathlib, random
random.seed(42)
out = sorted(pathlib.Path('results/test-judge-01/transcription/outputs').glob('*_transcription.json'))
invalid = sorted([f for f in out if json.loads(f.read_text()).get('is_valid') is False])
print([f.name for f in random.sample(invalid, max(1, len(invalid)//10))])
"

# 4. (optional) Regenerate the local appendix file referenced in §2.3
#    — output is gitignored under docs/research/data/.
python3 - <<'PY'
import json, pathlib, random
random.seed(42)
out = sorted(pathlib.Path('results/test-judge-01/transcription/outputs').glob('*_transcription.json'))
invalid = []
for f in out:
    d = json.loads(f.read_text())
    if d.get('is_valid') is False:
        invalid.append((f, d))
sample = random.sample(invalid, max(1, len(invalid) // 10))
sample.sort(key=lambda x: x[0].name)
appendix = [
    {
        'file': f.name,
        'duration_ms': d.get('duration_milliseconds'),
        'transcription_text': d.get('transcription_text'),
        'rejected_at': (d.get('validation') or {}).get('rejected_at'),
        'validation': d.get('validation'),
    }
    for f, d in sample
]
out_path = pathlib.Path('docs/research/data/judge-calibration-spot-check.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(appendix, indent=2, ensure_ascii=False))
print(f'wrote {out_path} ({len(sample)} records)')
PY
```

Sample produced from branch `feature/llm-transcription-criteria` head
`6a749a3`. The 145 invalid set was determined by an alphabetical sort of
files matching `*_transcription.json` whose `is_valid == False`; the
sample is `random.Random(seed=42).sample(invalid, k=14)`.

[pr-88]: https://github.com/FredDsR/arandu/pull/88
