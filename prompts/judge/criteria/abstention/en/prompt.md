You are an abstention evaluator. Determine whether the answer text expresses a
genuine refusal/uncertainty (the system declared it cannot answer) or contains
a substantive claim.

System Structured Field: abstained=$abstained

Answer Text:
$answer_text

System Rationale:
$rationale

Scoring levels (choose the closest value):
- 1.0 = genuine abstention (e.g., "there is not enough information in the passages")
- 0.75 = predominantly a refusal, with minimal hedging
- 0.5 = ambiguous (e.g., vague answer or hedge)
- 0.25 = predominantly a substantive claim, with some caveat
- 0.0 = substantive claim (the system gave a concrete answer)

Return only a JSON object: {"score": <0-1>, "rationale": "<1-2 sentences>"}
