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

Return only a JSON object: {"score": <0-1>, "rationale": "<1-2 sentences>"}
