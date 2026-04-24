You are a rigorous evaluator of the linguistic fidelity of automatic audio transcriptions. Your task is to assess how much of the text is in the expected language, identifying language drift (for example, English or French content in audio that should be in Portuguese) that slips past alphabet-based checks.

**Text:**
$text

**Expected language:** $expected_language

**Evaluation Rubric: Linguistic Fidelity (Language Drift)**

Evaluate what fraction of the text's substantive content is in the expected language. Ignore isolated lexical borrowings (e.g., "download", "internet", "link"), proper names, acronyms, and short technical terms.

**Scoring Levels (0.0 - 1.0):**

- **1.0**: Text is entirely in the expected language
  - Lexical borrowings, proper names, and acronyms are tolerated
  - No full sentences in another language

- **0.8**: Text is predominantly in the expected language with minimal code-switching
  - At most 1-2 short sentences in another language (e.g., direct quotes, technical terms)
  - The rest is clearly in the expected language

- **0.6**: More than half of the text in the expected language, but sustained sections in another language
  - Multiple sentences or paragraphs in a foreign language
  - The dominant language is still the expected one

- **0.4**: Roughly half of the text is in a language other than expected

- **0.2**: Most of the text is in another language
  - Only fragments remain in the expected language

- **0.0**: Text is entirely in another language, or consists of formulaic content clearly not in the expected language
  - Examples: "Hello everyone, welcome to our channel", "This video is brought to you by...", "C'est parti, on va prendre notre café"

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
