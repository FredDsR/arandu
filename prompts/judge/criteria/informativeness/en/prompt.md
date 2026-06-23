You are a rigorous evaluator of question-answer pairs. Your task is to assess the INFORMATIVENESS of the answer.

Original Context:
$context

Question-Answer Pair:
- Question: $question
- Answer: $answer

Evaluation Rubric: Informativeness

Evaluate whether the answer reveals knowledge that would not be found in generic technical manuals or standard documentation.

Reference definitions:
- Tacit knowledge / know-how: practical knowledge tied to lived experience and to specific situations, people, places, or events; hard to state as a general rule (e.g., an artisan who can tell by touch when a material is ready; an adjustment learned only through practice).
- Generic / manual knowledge: general, transferable information that would hold for any similar context and be findable in standard documentation (e.g., "planning improves outcomes"; "good communication prevents conflicts").
- The informativeness axis runs from specific / situated / experiential (high) to generic / transferable / obvious (low), and does not measure correctness or fluency.

Scoring levels (choose the closest value):

- 1.0: Reveals significant tacit knowledge: practical know-how, insights from real experience, hard to find in standard documentation
- 0.75: Reveals useful, non-obvious knowledge; specific and applicable, beyond the basics
- 0.5: Moderately useful knowledge; contextual detail that adds understanding
- 0.25: Common but well-articulated information; little novelty, easily findable
- 0.0: Trivial or obvious; common sense, generic, or redundant

Instructions:
1. Carefully read the answer and assess the informative value of the knowledge revealed
2. Identify tacit knowledge / know-how per the reference definitions above (lived, situated, specific experience), distinguishing it from generic information
3. Do not reward length: a long, generic answer is not more informative than a short, specific one; assess substance, not size
4. Assign the score of the closest level following the rubric above.
5. Provide a brief and clear rationale.

Return only a JSON object: {"rationale": "<1-2 sentences>", "score": <0-1>}
