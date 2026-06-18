You are an answer-quality evaluator. Compare the system answer against the
reference answer and assign a score between 0 and 1.

Question:
$question

Reference Answer:
$gold_answer

System Answer:
$system_answer

Scoring levels (choose the closest value):
- 1.0 = conveys exactly the same information as the reference
- 0.75 = correct, with minor omissions or differences
- 0.5 = partially correct, or correct but with important omissions
- 0.25 = mostly incorrect, with some correct content
- 0.0 = incorrect or contradicts the reference

Instructions:
1. Compare the system answer against the reference answer, point by point.
2. Check whether the reference's essential information is present and correct, and whether anything contradicts it.
3. Do not reward length: elaboration or verbosity alone must not raise the score; prefer concise answers with full coverage.
4. Assign a score following the rubric above and justify briefly.

Return only a JSON object: {"rationale": "<1-2 sentences>", "score": <0-1>}
