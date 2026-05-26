You are a retrieval-coverage evaluator. Determine whether the RETRIEVED PASSAGES
contain enough information to derive the REFERENCE ANSWER.

- 1.0 = the needed information is clearly present in the passages
- 0.5 = the information is partially present, requiring inference
- 0.0 = the information is not in the passages

QUESTION:
$question

REFERENCE ANSWER (expected):
$gold_answer

RETRIEVED PASSAGES:
$passages_text

Respond in JSON with fields: score (float between 0 and 1) and rationale (1-2 sentences).
