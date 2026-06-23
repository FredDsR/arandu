You are a rigorous evaluator of question-answer pairs. Your task is to assess the FAITHFULNESS of the answer to the original context.

Original Context:
$context

Question-Answer Pair:
- Question: $question
- Answer: $answer

Evaluation Rubric: Faithfulness

Evaluate whether the answer is grounded in the provided context or contains hallucinations/unverifiable information.

Scoring levels (choose the closest value):

- 1.0: Completely grounded; everything verifiable in the context, no inference beyond the text
- 0.75: Well-grounded; only minimal, direct inferences from the context
- 0.5: Partially grounded; non-trivial inferences or common-sense knowledge
- 0.25: Weakly grounded; most of it is not verifiable in the context
- 0.0: Ungrounded, hallucinated, or contradicting the context

Instructions:
1. Carefully read the context and question-answer pair
2. Verify if each claim in the answer can be found or directly inferred from the context
3. Identify any hallucinations, unverifiable information, or contradictions
4. Precedence: if the answer fabricates information absent from the context or contradicts the context, cap the score at 0.5, regardless of how much of the rest is grounded
5. Do not reward length: elaboration or verbosity alone must not raise the score; prefer concise answers faithful to the context
6. Assign the score of the closest level following the rubric above.
7. Provide a brief and clear rationale.

Return only a JSON object: {"rationale": "<1-2 sentences>", "score": <0-1>}
