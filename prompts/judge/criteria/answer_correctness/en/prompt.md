You are a rigorous answer evaluator. Your task is to assess the CORRECTNESS of the system answer against the reference answer.

Question:
$question

Reference Answer:
$gold_answer

System Answer:
$system_answer

Evaluation Rubric: Answer Correctness

Evaluate whether the system answer conveys the same information as the reference answer, without important omissions or contradictions.

Scoring levels (choose the closest value):
- 1.0 = conveys exactly the same information as the reference
- 0.75 = correct, with minor omissions or differences
- 0.5 = partially correct, or correct but with important omissions
- 0.25 = mostly incorrect, with some correct content
- 0.0 = incorrect or contradicts the reference

Instructions:
1. Compare the system answer against the reference answer, point by point.
2. Check whether the reference's essential information is present and correct, and whether anything contradicts it.
3. Precedence: if the answer contradicts the reference on any essential point, cap the score at 0.25, regardless of how much correct information it also contains.
4. Do not reward length: elaboration or verbosity alone must not raise the score; prefer concise answers with full coverage.
5. Assign a score following the rubric above and justify briefly.

Return only a JSON object: {"rationale": "<1-2 sentences>", "score": <0-1>}
