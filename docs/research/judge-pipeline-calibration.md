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
pipeline. A dual-class audit was conducted: a 30 % sample of the rejection
set (n = 43) and a 15 % sample of the admission set (n = 31), both drawn
deterministically with `random.seed(42)` (75 records total, 21 % of the
corpus). Of the 43 audited rejections, 38 are unambiguous true positives
covering Whisper silence-fillers, single-word- and phrase-level repetition
loops, language-drift hallucinations, and real-content recordings
contaminated by artificial tails; the remaining 5 are borderline
rejections at the `hallucination_loop` decision boundary (score 0.6 vs
threshold 0.7), where the validator flagged mild formulaic repetition in
otherwise-plausible interview content. No unambiguous false positives
were observed. Of the 31 audited admissions, 30 are unambiguous true
negatives and 1 is a false negative — a "Thank you." Whisper
silence-filler that slipped past the heuristic `content_density`
threshold (score ≈ 0.475 vs threshold 0.40) and was scored 1.0 by both
LLM criteria due to a "very short text is not evaluable" deference rule
in their prompts. Sample false-positive rate is 0 % under the lenient
reading and 11.6 % under the conservative reading; sample false-negative
rate is 3.2 %. The two pipeline stages contribute complementary coverage
(58 % of rejections by heuristics, 42 % by the LLM stage) and the
overall result supports retaining both stages and the current default
thresholds for the riverine-fieldwork corpus. The single false
negative motivated a `content_length_floor` heuristic now installed at
the head of the heuristic stage (§ 4.5), which rejects the audited
record by construction and subsumes the legacy 200-character filter
previously embedded in the QA-generation step.

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
| `content_length_floor` threshold | `0.5` (binary; 200 chars / 30 words) — **added post-audit, see § 4.5** |
| Hardware | NVIDIA RTX 4090 (24 GB), tupi6 |
| Wall clock | 75 m (1.27 s/record amortised) |

### 2.3 Audit protocol

A dual-class audit was performed: 30 % of the rejection set was sampled
to bound the false-positive rate, and 15 % of the admission set was
sampled in a separate draw to bound the false-negative rate. Both
samples were drawn deterministically using `random.Random(seed=42)`
applied successively to the alphabetically sorted lists of rejected
and admitted files:

```python
random.seed(42)
rejected_sample  = random.sample(rejected_records, k=⌊145 * 0.30⌋ = 43)
admitted_sample  = random.sample(admitted_records, k=⌊208 * 0.15⌋ = 31)
```

The seed is fixed to make the audit reproducible. Drawing both samples
in a single seeded session (rejected first, admitted second) ensures
the script in §7 reproduces both byte-identically.

For each rejected record the reviewer inspected:

- transcription text content (full text, not just preview);
- duration in milliseconds;
- pipeline rejection stage (`heuristic_filter` vs `llm_filter`);
- failing criterion names, scores, thresholds, and full rationales;

and classified the rejection as **true positive** (TP — the rejection
is well-founded), **false positive** (FP — the record is a real-content
transcription wrongly rejected), or **borderline** (the validator
surfaced a real artefact but a defensible counter-reading exists,
typically at scores within 0.1 of the threshold).

For each admitted record the reviewer inspected the full transcription
text and classified the admission as **true negative** (TN — the
record is real interview content) or **false negative** (FN — a Whisper
failure mode the pipeline should have caught). Particular attention was
paid to text heads and tails (the regimes where contaminated-tail
hallucinations and silence-filler stubs concentrate), and to records
near the heuristic decision boundaries.

The complete samples (43 + 31 = 74 records, including raw transcription
text and full `JudgePipelineResult` payloads) can be regenerated locally
with the script in §7 and are byte-identical to the corresponding
records on disk. Raw transcription text contains participant names from
the fieldwork recordings and is therefore kept out of version control
under the project-wide convention applied to `results/`; the
regeneration recipe in §7 produces both appendix files deterministically
from a run's output directory.

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

### 3.3 Rejection audit — manual classification

Of 43 audited rejections, 38 are unambiguous true positives and 5 are
borderline cases at the `hallucination_loop` decision boundary
(discussed in § 4.4). No unambiguous false positives were observed.
Categories:

| Category | n | Stage | Representative behaviour |
|---|---|---|---|
| Whisper silence-filler (very short stub in seconds-to-tens-of-seconds audio) | 7 | mostly heuristic | "Hello, everyone." (8.7 wpm); "Thank you." × 4 (9–11 wpm, three rejected by `content_density`, one by `language_drift`); "Gracias. Gracias." (2.4 wpm); "We'll be right back." (8.4 wpm) |
| Pure repetition loops (single phrase × N) | 9 | heuristic | "Vamos lá" × 148; "eu quero branco" × 87; "Hello everyone!" × 10; "quatro" × 221; "tchau" × 148; "Viva! Viva!" × 17; "It's time to get out of here" × 7; "a little bit of a little bit" × 110; "welcome back to my channel" × 11 |
| English drift / Whisper closing-line hallucination | 5 | mostly llm | "I don't know. I'll see you next time"; "I'll see you next time"; "It's time to get out of here" × 3; "I'm going to take a look at this one. I'll see you next time"; "Hello everyone, welcome back to our channel" |
| Empty / whitespace-only transcription | 1 | heuristic | text == "." → `script_match = 0.5` (`no_alphabetic_content`) + low density |
| Cyrillic credits text | 1 | heuristic | "Редактор субтитров А.Семкин Корректор А.Егорова" (Whisper hallucinated Russian video credits) |
| Sustained French drift with internal hallucination loops | 1 | llm | predominantly French with `language_drift = 0.2` and "C'est ici" repetition runs |
| Real PT interview with hallucination tail or interspersed loop | 9 | mixed | meditation-script tokens + "1900, 1900, 1900"; "tchau, tchau" × 146-148 tails on real interviews (3 cases); disconnected "excluída do bebê do bebê" run; "É, é, é" + "não, tranquilo, tranquilo" runs; formulaic phrases imitating script structure; "Eu ia dar um furo gigantesco" × 6; "A A A A A A" tail; phrase loops "o que é o que é o" × 137-148 |
| Real-content English passage with internal repetition loop | 1 | heuristic | English narrative with "it's it's it's" × 210, "something, something" × 208 |
| Real PT interview, `hallucination_loop = 0.6` (borderline) | 5 | llm | category-list discussion; long family/violence narrative; traditional-medicine narrative; flooding-solutions discussion; flood-relief enumeration ("Distribuir o telha, Distribuir o prego") — see § 4.4 |
| **Total audited** | 43 | — | — |

### 3.4 Admission audit — false-negative classification

Of 31 audited admissions, 30 are unambiguous true negatives (real
Portuguese interview content of varying lengths). One is a false
negative:

| File | Duration | Text | Why it slipped |
|---|---|---|---|
| `12Oi6lbrKtwS-_YBdB5C0IqRizQIgpbl1_transcription.json` | 8.4 s | "Thank you." (10 chars, 2 words) | `content_density` score = 0.475 (just above the 0.40 threshold); `language_drift = 1.0` and `hallucination_loop = 1.0` because both prompts defer when text is "very short or not evaluable" |

This is the same kind of Whisper silence-filler that the four
`Thank you.` rejections in the rejection sample (§ 3.3) caught at
9–11 wpm; the only difference is that 14.2 wpm at 8.4 s passes the
linearly-scaled `content_density` threshold by 0.075 of a point, after
which the LLM stage explicitly defers. The mechanism is discussed in
§ 4.5.

### 3.5 Sample rationale strings

Verbatim rationales surfaced by the LLM criteria are reproduced below
(the validator model writes in Portuguese):

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

### 4.1 Sample-based bounds on the error rates

The rejection-side audit (n = 43, 0 unambiguous FP) bounds the
false-positive rate in the rejection set:

- **Lenient reading** (the 5 borderline cases are correct rejections,
  consistent with the validator's own rationales): 0 FP in 43.
  Rule-of-three upper 95 % bound on the population FP rate: 3 / 43
  ≈ 7.0 %, i.e. ≤ ~ 10 of 145 rejected records.
- **Conservative reading** (all 5 borderline cases reclassified as
  FP): 5 FP in 43, point estimate 11.6 %, exact (Clopper–Pearson)
  95 % CI [3.9 %, 25.1 %], i.e. ≤ ~ 36 of 145 rejected records under
  the upper bound.

The admission-side audit (n = 31, 1 FN) bounds the false-negative rate
in the admission set:

- 1 FN in 31, point estimate 3.2 %, exact 95 % CI [0.1 %, 16.7 %],
  i.e. ≤ ~ 35 of 208 admitted records under the upper bound.

Combined, the audit data are consistent with the pipeline operating
correctly on the bulk of the corpus and exposing one specific
calibration gap (§ 4.5) that affects very-short Whisper silence-fillers
admitted via a narrow band of `content_density` scores.

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

Two regimes worth flagging surface in the 43-record rejection audit.

**Contaminated-tail records.** Multiple records contain mostly-real PT
interview content concluded by, or interspersed with, an artificial
repetition pattern. The audit surfaced 9 such cases — for example
`1Vhf1EijZ6Rn...` ("Eu ia dar um furo gigantesco" × 6 in real
road-reconstruction interview); `1vBRB2iFq92v...` ("A A A A A A" tail
in real flooding interview); `1NRECauGexuyl7...` and `1eWFFqjeAbMZbke1...`
and `1N7sET0CNni_d6bOlwi_U...` ("tchau, tchau, tchau" × 146-148 tails
on otherwise-real interviews); `10q1gaI_wnyqQUU...` (meditation-script
fragments and "1900, 1900, 1900" runs); `1bYIPOewPqsx9Ys8...` ("É, é,
é, é" + "não, tranquilo, tranquilo" runs); `1zk-N6Dvek9zQgu9...`
(disconnected "excluída do bebê do bebê" runs); `1Knaaeo4O4KxbgPU...`
("o que é o que é o" × 137-148 phrase loop in a real interview).
For the research goal (Knowledge-Graph extraction over the riverine
community's tacit knowledge) the pipeline's conservative whole-record
rejection is desirable: a contaminated tail propagates fabricated
entities into the KG, and the downstream extraction step has no
mechanism to localise the contamination within the text. A future
refinement could mark such records as "recoverable" and pass only the
clean prefix to extraction; that optimisation is out of scope here.

**Borderline `hallucination_loop` rejections.** Five records
(`15DCJlZR3jJ3...`, `1L1_XL8UWqxZgi...`, `1MGroACSWxP21...`,
`1oAMvTu3GdsepODg...`, `1rAwXptR47e-_nU...`) sit at
`hallucination_loop = 0.6`, i.e. 0.1 below the 0.7 threshold. The
validator's own rationales acknowledge that the bulk of the content is
plausible and identify only mild formulaic patterns: category-list
discussions (research methodology), long family-violence narratives
with light repetition for emphasis, traditional-medicine narratives with
"sim, sim, sim" / "não, não, não" affirmation runs, flooding-solutions
discussions with "E eu vou falar..." short repeats, and flood-relief
material enumerations ("Distribuir o telha, Distribuir o prego, ..."),
the last of which is plausibly a real interviewee enumeration mis-read
as a list-template hallucination.

The cluster of five 0.6-band cases is informative: they are not random
near-misses but a coherent class — real Portuguese interviews that
contain real-content repetition (affirmation runs, lists, returning to
a topic) which the validator over-attributes to the formulaic-loop
pattern. Two interpretations are defensible:

- The conservative reading treats the rejections as correct (the
  validator did surface a real artefact pattern; the threshold is
  doing its job, and KG-extraction quality is best protected by
  whole-record rejection).
- The lenient reading treats them as false positives and motivates
  either raising the prompt's tolerance for natural enumeration or
  lowering the threshold from 0.7 to 0.55, which would admit all five
  at the cost of also admitting any record with comparable signal
  strength.

The audit data alone do not resolve the choice. A second-rater review
or an extended audit at the boundary (records scoring 0.5–0.7 on
`hallucination_loop`) would be the principled next step; until then
the default behaviour favours conservative rejection, consistent with
the KG-protection rationale above.

### 4.5 Calibration gap at the silence-filler boundary

The single false negative observed in the admission audit (§ 3.4) is
informative: the same "Thank you." Whisper silence-filler that the
pipeline rejected four times in the rejection sample slipped through
once because its specific duration / word-count combination produced a
`content_density` score of 0.475 — just above the 0.40 threshold. The
LLM stage then deferred (`language_drift` and `hallucination_loop`
both scored 1.0) because both prompts contain a "very short text →
score 1.0 (not evaluable)" rule:

```
5. Texto vazio ou muito curto deve receber nota 1.0 (não avaliável).
```

The rule was added defensively so the LLM does not penalise legitimately
short utterances, but in combination with `content_density`'s linear
scaling it leaves a narrow admission band: a record short enough to
reach the LLM stage's deference rule but with a wpm that puts
`content_density` above the cutoff. For the audited record this band is
12–14 words/min at a 6–9 second duration; it is plausible that a
handful of similar records exist in the unaudited admission set.

Three minimally-invasive remediations were considered:

1. **A non-scaled length-floor heuristic.** Add a binary
   `content_length_floor` criterion at the front of the heuristic
   stage that rejects records below a minimum character and word
   count regardless of duration; this short-circuits the silence-filler
   band before any other check runs. **Adopted** — implemented as
   `arandu.transcription.criteria.ContentLengthFloorCriterion`
   (defaults: 200 chars OR 30 words; both knobs configurable). The
   audited false negative ("Thank you.", 10 chars / 2 words) fails
   this floor by construction, and the gate also subsumes the previous
   200-char `MIN_CONTEXT_LENGTH` filter that lived inside the QA
   generation step (now removed; the QA layer trusts upstream
   judging).
2. Lift `content_density`'s minimum-wpm threshold from 30 wpm towards
   40 wpm; closes the 12–14 wpm band but also rejects some legitimate
   single-utterance interviews. **Not adopted** — option 1 closes
   the gap without affecting the wpm-only regime.
3. Replace the LLM "too short → 1.0" rule with a "too short → defer
   to a stricter heuristic language check" branch. **Not adopted** —
   becomes redundant once option 1 prevents short records from
   reaching the LLM stage, but worth revisiting if a future record
   class slips past option 1.

The implementation closes the silence-filler boundary specifically;
records that satisfy both floors but otherwise look like
hallucinations remain the LLM stage's responsibility, and the §4.4
borderline-cluster question (real-content interviews near
`hallucination_loop = 0.6`) is unaffected and remains open.

## 5. Limitations

1. **Sample size.** The dual-class audit (43 rejected + 31 admitted =
   74 records, 21 % of the corpus) yields rule-of-three upper 95 %
   bounds of 7.0 % on the unambiguous false-positive rate and an exact
   95 % upper bound of 16.7 % on the false-negative rate (1 / 31
   observed). Tighter regimes (point-estimate FP / FN below 2 %) would
   require either a substantially larger audit (n ≥ 150 each side,
   approaching full enumeration of a 353-record corpus) or a corpus
   substantially larger than 353 records. The current bounds are
   adequate to support the calibration claim and to motivate the
   specific remediation in § 4.5; they are not adequate to certify a
   below-2 % error regime in either direction.
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
5. **Negative-class audit limited to surface inspection.** The 31
   admitted records were classified primarily by reading the full
   transcription text and noting any obvious Whisper failure modes
   (drift, repetition runs, contaminated tails, formulaic stubs). A
   stricter protocol would also play back the original audio for
   sentence-level alignment checks; this would surface low-SNR
   invention or name/number substitutions that are invisible from
   text alone (cf. the limitations called out in the
   `TranscriptionJudge` docstring). Such a protocol is feasible but
   was out of scope for this audit.

## 6. Conclusion

The transcription judge pipeline rejects 41.1 % of the `test-judge-01`
corpus, with the rejection split 58 / 42 between the heuristic and LLM
stages. A dual-class audit of 43 rejected and 31 admitted records
(21 % of the corpus, drawn deterministically with `seed=42`) found:

- **0 unambiguous false positives** in the rejection sample;
- **5 borderline rejections** at `hallucination_loop = 0.6` —
  real-content interviews where the validator surfaced mild formulaic
  patterns; rule-of-three upper 95 % bound on the FP rate is 7.0 %
  under the lenient reading and 25.1 % under the conservative
  reclassification of all 5 borderline cases;
- **1 false negative** in the admission sample — a "Thank you."
  Whisper silence-filler that slipped through a narrow band of
  `content_density` scores combined with the LLM stage's
  too-short-to-evaluate deference rule (point estimate 3.2 %, exact
  95 % upper bound 16.7 %).

The two pipeline stages contribute complementary coverage: heuristics
catch repetition loops, low-density stubs, wrong-script content, and
high-density runs (58 % of all rejections); the LLM stage catches
sustained Latin-script language drift and short formulaic
hallucinations the heuristics admit (42 % of all rejections). The
silence-filler calibration gap surfaced by the admission audit was
closed in-thesis by adding a `content_length_floor` heuristic at the
front of the pipeline (§ 4.5); the legacy 200-character QA-generation
filter was removed in the same change so the responsibility lives in
exactly one place. The combined two-stage pipeline can be recommended
for routine use on the dissertation corpus at the current default
thresholds, with one open question about the 0.6-band borderline
behaviour (§ 4.4) deferred to follow-up work.

## 7. Reproducibility

```bash
# 1. Reproduce the run on the cluster:
PIPELINE_ID=test-judge-01 sbatch scripts/slurm/judge/transcription/tupi.slurm

# 2. Pull verdicts back to the local checkout:
rsync -avz fdsreckziegel@pcad.inf.ufrgs.br:~/etno-kgc-preprocessing/results/test-judge-01/transcription/outputs/ \
  results/test-judge-01/transcription/outputs/

# 3. Regenerate both audit appendices in one seeded session.
#    Outputs are gitignored under docs/research/data/.
python3 - <<'PY'
import json, pathlib, random

random.seed(42)
out_dir = pathlib.Path('results/test-judge-01/transcription/outputs')
records = sorted(out_dir.glob('*_transcription.json'))

valid_records, invalid_records = [], []
for f in records:
    d = json.loads(f.read_text())
    if d.get('is_valid') is True:
        valid_records.append((f, d))
    elif d.get('is_valid') is False:
        invalid_records.append((f, d))

# Same single seeded session used to produce the audited samples.
rejected_sample = random.sample(invalid_records, max(1, len(invalid_records) * 30 // 100))
admitted_sample = random.sample(valid_records,   max(1, len(valid_records)   * 15 // 100))
rejected_sample.sort(key=lambda x: x[0].name)
admitted_sample.sort(key=lambda x: x[0].name)

def dump(sample, path):
    payload = [
        {
            'file': f.name,
            'duration_ms': d.get('duration_milliseconds'),
            'transcription_text': d.get('transcription_text'),
            'is_valid': d.get('is_valid'),
            'rejected_at': (d.get('validation') or {}).get('rejected_at'),
            'validation': d.get('validation'),
        }
        for f, d in sample
    ]
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

dump(rejected_sample, 'docs/research/data/judge-calibration-rejected-30pct.json')
dump(admitted_sample, 'docs/research/data/judge-calibration-admitted-15pct.json')
print(f'rejected sample: {len(rejected_sample)} of {len(invalid_records)}')
print(f'admitted sample: {len(admitted_sample)} of {len(valid_records)}')
PY
```

Samples produced from branch `feature/llm-transcription-criteria` head
`6a749a3`. The 145 rejected and 208 admitted sets were determined by an
alphabetical sort of files matching `*_transcription.json` whose
`is_valid == False` and `is_valid == True` respectively; the samples
are
`random.Random(seed=42).sample(rejected, k=43)` followed in the same
session by `random.sample(admitted, k=31)`.

[pr-88]: https://github.com/FredDsR/arandu/pull/88
