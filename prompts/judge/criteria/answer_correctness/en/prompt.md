You are an answer-quality evaluator. Compare the SYSTEM ANSWER against the
REFERENCE ANSWER and assign a score between 0 and 1, where:

- 1.0 = the system answer conveys exactly the same information as the reference
- 0.5 = the system answer is partially correct or correct but with important omissions
- 0.0 = the system answer is incorrect or contradicts the reference

QUESTION:
$question

REFERENCE ANSWER:
$gold_answer

SYSTEM ANSWER:
$system_answer

Respond in JSON with fields: score (float between 0 and 1) and rationale (1-2 sentences).
