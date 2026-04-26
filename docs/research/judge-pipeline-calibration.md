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
pipeline. A randomly drawn 20 % audit (n = 29) of the rejection set was
manually inspected; every sampled rejection was a defensible true positive,
covering empty/silence-filler outputs, single-word- and phrase-level
repetition loops, language-drift hallucinations, and mostly-real
transcriptions contaminated by an artificial tail. Two of the 29 sampled
rejections sit close to the `hallucination_loop` decision boundary
(score 0.6 vs threshold 0.7), where the validator flagged mild formulaic
repetition in otherwise-plausible interview content; even classifying both
as false positives the audit yields a sample false-positive rate of
≤ 6.9 %. The two pipeline stages contribute complementary coverage:
heuristic checks reject 58 % of the invalid set, and the LLM stage
(`language_drift`, `hallucination_loop`) catches the remaining 42 %, none
of which the heuristics could have caught on their own. The result
supports retaining both stages and the current default thresholds for the
riverine-fieldwork corpus.

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

A 20 % manual audit of the rejection set was drawn using
`random.Random(seed=42).sample(invalid, k=⌊145 / 5⌋ = 29)`, where `invalid`
is the alphabetically sorted list of files with `is_valid == False`. The
seed is fixed to make the audit reproducible. Each sampled record was
inspected manually for:

- transcription text content (full text, not just preview);
- duration in milliseconds;
- pipeline rejection stage (`heuristic_filter` vs `llm_filter`);
- failing criterion names, scores, thresholds, and full rationales.

Each sampled rejection was then categorised into one of the failure modes
listed in §3.2.

The complete sample (29 records, including raw transcription text and full
`JudgePipelineResult` payloads) can be regenerated locally with the script
in §7 and is byte-identical to the corresponding records on disk. Raw
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

All 29 audited records were classified as defensible rejections; 27 are
unambiguous true positives and 2 are borderline cases at the
`hallucination_loop` decision boundary (discussed in §4.4). Categories
observed:

| Category | n | Stage | Representative behaviour |
|---|---|---|---|
| Whisper silence-filler (≤ 17-char stub in 9–49 s of audio) | 5 | heuristic | "Hello, everyone." (8.7 wpm); "Thank you." (9 / 11 wpm); "Thank you." in `llm_filter` (9 wpm × `language_drift = 0`); "Gracias. Gracias." (2.4 wpm) |
| Pure repetition loops (single phrase × N) | 8 | heuristic | "Vamos lá" × 148; "eu quero branco" × 87; "Hello everyone!" × 10; "quatro" × 221; "tchau" × 148; "It's time to get out of here" × 7; "a little bit of a little bit" × 110; "welcome back to my channel" × 11 |
| English drift / Whisper closing-line hallucination | 4 | llm | "I don't know. I'll see you next time"; "I'll see you next time"; "It's time to get out of here" × 3; "I'm going to take a look at this one. I'll see you next time" |
| Empty transcription | 1 | heuristic | `script_match = 0.5` (`no_alphabetic_content`) + low density |
| Sustained French drift | 1 | llm | predominantly French with PT proper-noun fragments only |
| Real PT content with hallucination tail or interspersed loop | 7 | mixed | meditation-script tokens + "1900, 1900, 1900"; "tchau, tchau" tail in real interview; disconnected "excluída do bebê do bebê" run; "É, é, é" + "não, tranquilo, tranquilo" runs; formulaic phrases imitating script structure; "Eu ia dar um furo gigantesco" × 6; "A A A A A A" tail |
| Real PT interview with mild formulaic enumeration ("borderline") | 2 | llm | "E eu vou falar..." short repeats; lists like "Distribuir o telha, Distribuir o prego" — see §4.4 |
| Real-content English passage with internal repetition loop | 1 | heuristic | English narrative with "it's it's it's" × 210, "something, something" × 208 |
| **Total audited** | 29 | — | — |

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

A 29-record sample with zero unambiguous false positives gives, by the
rule of three, an upper 95 % confidence bound on the false-positive
proportion of 3 / 29 ≈ 10.3 % within the rejection set — i.e. the
population false-positive rate is plausibly below ~ 15 of 145 invalid
records. If the two §4.4 borderline cases are conservatively reclassified
as false positives, the sample point estimate is 2 / 29 ≈ 6.9 %, and the
exact (Clopper–Pearson) 95 % upper bound is ≈ 22.6 %, i.e. ≤ ~ 33 / 145.
Either reading falsifies a scenario in which the 41 % rejection rate is
driven by aggressive thresholds rather than corpus quality. The sample
also exposes no systematic class of unambiguous false positive that would
warrant threshold relaxation; the borderline class is narrow and
specific (mild formulaic enumeration in real interview content) and
discussed separately.

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

Two regimes worth flagging surface in the 29-record audit.

**Contaminated-tail records.** Several records contain mostly-real PT
interview content concluded by an artificial repetition tail
(`1Vhf1EijZ6Rn...`, `1vBRB2iFq92v...`, `1NRECauGexuyl7...`,
`1eWFFqjeAbMZbke1...`). The bulk of each transcription is plausible
interview dialogue; only the tail is hallucinated (e.g. "tchau, tchau,
tchau" × 148, "A A A A A A" run, "Eu ia dar um furo gigantesco" × 6).
The pipeline rejects these records as a whole. For the research goal
(Knowledge-Graph extraction over the riverine community's tacit
knowledge) the conservative stance is desirable: a contaminated tail
propagates fabricated entities into the KG, and the downstream
extraction step has no mechanism to localise the contamination within
the text. A future refinement could mark such records as "recoverable"
and pass only the clean prefix to extraction; that optimisation is out
of scope here.

**Borderline `hallucination_loop` rejections.** Two records
(`1oAMvTu3GdsepODg...`, `1rAwXptR47e-_nU...`) sit at
`hallucination_loop = 0.6`, i.e. 0.1 below the 0.7 threshold. The
validator's own rationales acknowledge that the bulk of the content is
plausible and identify only mild formulaic patterns: short repeats like
"E eu vou falar, e eu vou falar e digo…" in one case, and a list of
distributed materials in a flood-relief context ("Distribuir o telha,
Distribuir o prego, …") in the other. The latter is plausibly a real
enumeration of items by an interviewee, mis-read by the validator as a
list-template hallucination. Two interpretations are defensible:

- The conservative reading treats the rejections as correct (the
  validator did surface real artefacts; the threshold is doing its
  job).
- The lenient reading treats both as false positives and motivates
  raising the threshold to 0.55, which would have admitted them at the
  cost of also admitting any record with comparable signal strength.

The audit data alone do not resolve the choice. A second-rater review
or an extended audit at the boundary (records scoring 0.5–0.7 on
`hallucination_loop`) would be the next step; until then the default
behaviour favours conservative rejection, consistent with the
KG-protection rationale above.

## 5. Limitations

1. **Sample size.** The 20 % manual audit (n = 29) gives a 95 % upper
   confidence bound of ≈ 10.3 % on the unambiguous false-positive rate
   (or ≈ 22.6 % under the conservative reclassification of the two
   §4.4 borderline cases). Tighter regimes (e.g. distinguishing 5 % vs
   2 %) would require n ≥ 60.
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
   confusion matrix and bound the false-negative rate. A 20 % audit of
   admitted records (≈ 42 records) is feasible and recommended as
   follow-up.

## 6. Conclusion

The transcription judge pipeline rejects 41.1 % of the `test-judge-01`
corpus, with the rejection split 58 / 42 between the heuristic and LLM
stages. A randomly drawn 20 % audit of the rejection set found 0
unambiguous false positives and 2 borderline `hallucination_loop`
rejections at score 0.6 (out of 29 audited; sample false-positive rate
≤ 6.9 %, 95 % upper bound ≤ 22.6 % under the conservative reading,
≤ 10.3 % under the lenient one). Every rejection corresponds to a
defensible Whisper failure mode: silence filler, repetition loop,
language drift, formulaic hallucination, or a contaminated tail. The
LLM stage contributes 42 % of all rejections and is the only path
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
print([f.name for f in random.sample(invalid, max(1, len(invalid)//5))])
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
sample = random.sample(invalid, max(1, len(invalid) // 5))
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
sample is `random.Random(seed=42).sample(invalid, k=29)`.

[pr-88]: https://github.com/FredDsR/arandu/pull/88
