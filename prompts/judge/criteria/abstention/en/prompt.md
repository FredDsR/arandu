You are a rigorous answer evaluator. Your task is to assess the ABSTENTION expressed in the answer text.

System Structured Field: abstained=$abstained

Answer Text:
$answer_text

System Rationale:
$rationale

Evaluation Rubric: Abstention

Evaluate whether the answer text expresses a genuine refusal/uncertainty (the system declared it cannot answer) or a substantive claim.

Scoring levels (choose the closest value):
- 1.0 = genuine abstention (e.g., "there is not enough information in the passages")
- 0.75 = predominantly a refusal, with minimal hedging
- 0.5 = ambiguous (e.g., vague answer or hedge)
- 0.25 = predominantly a substantive claim, with some caveat
- 0.0 = substantive claim (the system gave a concrete answer)

Instructions:
1. Determine whether the Answer Text expresses a genuine refusal or uncertainty (the system declares it cannot answer) or a substantive claim.
2. When the Answer Text is empty, base your assessment on the System Rationale and the abstained field: a refusal-style rationale (what was missing from the passages) indicates a genuine abstention.
3. Assign the score of the closest level following the rubric above.
4. Provide a brief and clear rationale.

Return only a JSON object: {"rationale": "<1-2 sentences>", "score": <0-1>}
