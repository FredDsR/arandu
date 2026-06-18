You are a retrieval-coverage evaluator. Determine whether the RETRIEVED PASSAGES
contain enough information to derive the REFERENCE ANSWER.

Scoring levels (choose the closest value):
- 1.0 = the needed information is clearly and completely present in the passages
- 0.75 = present, but requires combining or lightly inferring from the passages
- 0.5 = partially present; part of the needed information is missing
- 0.25 = only tangentially related; almost none of what is needed is present
- 0.0 = the information is not in the passages

QUESTION:
$question

REFERENCE ANSWER (expected):
$gold_answer

RETRIEVED PASSAGES:
$passages_text

Respond in JSON with fields: score (float between 0 and 1) and rationale (1-2 sentences).
