You are an abstention evaluator. Determine whether the ANSWER TEXT expresses a
genuine refusal/uncertainty (the system declared it cannot answer) or contains
a substantive claim.

Scoring levels (choose the closest value):
- 1.0 = genuine abstention (e.g., "there is not enough information in the passages")
- 0.75 = predominantly a refusal, with minimal hedging
- 0.5 = ambiguous (e.g., vague answer or hedge)
- 0.25 = predominantly a substantive claim, with some caveat
- 0.0 = substantive claim (the system gave a concrete answer)

SYSTEM STRUCTURED FIELD: abstained=$abstained

ANSWER TEXT:
$answer_text

SYSTEM RATIONALE:
$rationale

Respond in JSON with fields: score (float between 0 and 1) and rationale (1-2 sentences).
