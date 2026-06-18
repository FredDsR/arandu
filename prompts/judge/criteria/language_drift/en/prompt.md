You are a rigorous evaluator of the linguistic fidelity of automatic audio transcriptions. Your task is to assess how much of the text is in the expected language, identifying language drift (for example, English or French content in audio that should be in Portuguese) that slips past alphabet-based checks.

**Text:**
$text

**Expected language:** $expected_language

**Evaluation Rubric: Linguistic Fidelity (Language Drift)**

Evaluate what fraction of the text's substantive content is in the expected language. Ignore isolated lexical borrowings (e.g., "download", "internet", "link"), proper names, acronyms, and short technical terms.

**Scoring levels (choose the closest value):**

- **1.0**: Entirely in the expected language (borrowings, proper names, acronyms tolerated; no full sentences in another language)
- **0.75**: Predominantly the expected language with minimal code-switching (at most 1-2 short sentences in another language)
- **0.5**: Majority in the expected language, but with sustained sections (multiple sentences/paragraphs) in another language
- **0.25**: Roughly half or less in the expected language
- **0.0**: Entirely in another language, or formulaic content clearly not in the expected language

**Instructions:**
1. Identify the dominant language of the text.
2. Compare against the expected language ($expected_language).
3. Ignore short lexical borrowings, proper names, acronyms, and technical terms.
4. Flag full sentences in another language as drift.
5. Empty or very short text should receive a score of 1.0 (not evaluable).

**Return ONLY a JSON object in the following format:**
```json
{
  "score": 0.0,
  "rationale": "Brief explanation of the assigned score"
}
```
