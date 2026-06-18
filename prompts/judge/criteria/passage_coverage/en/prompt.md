You are a retrieval-coverage evaluator. Determine whether the retrieved passages
contain enough information to derive the reference answer.

Question:
$question

Reference Answer (expected):
$gold_answer

Retrieved Passages:
$passages_text

Scoring levels (choose the closest value):
- 1.0 = the needed information is clearly and completely present in the passages
- 0.75 = present, but requires combining or lightly inferring from the passages
- 0.5 = partially present; part of the needed information is missing
- 0.25 = only tangentially related; almost none of what is needed is present
- 0.0 = the information is not in the passages

Instructions:
1. Read the reference answer and identify the information needed to derive it.
2. Check whether that information is present in the retrieved passages, even if it requires combining excerpts.
3. Judge the passages' coverage relative to the reference, not writing quality.
4. Assign a score following the rubric above and justify briefly.

Return only a JSON object: {"rationale": "<1-2 sentences>", "score": <0-1>}
