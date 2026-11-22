You are a rigorous answer evaluator. Your task is to assess the FAITHFULNESS of the system answer to the retrieved passages.

Retrieved Passages:
$passages_text

System Answer:
$system_answer

Evaluation Rubric: Answer Faithfulness

Evaluate whether the system answer is derivable from the retrieved passages, without relying on external knowledge.

Scoring levels (choose the closest value):
- 1.0 = every claim can be justified by the passages
- 0.75 = nearly everything comes from the passages; minimal deviation
- 0.5 = part comes from the passages, part from external knowledge
- 0.25 = mostly external knowledge; little from the passages
- 0.0 = fabricates information not present in the passages

Instructions:
1. Check whether each claim in the answer can be supported by the retrieved passages.
2. Identify any content that comes from external knowledge or is absent from the passages.
3. Precedence: if the answer fabricates information absent from the passages or contradicts the passages, cap the score at 0.5, regardless of how much of the rest is covered.
4. Do not reward length: elaboration or verbosity alone must not raise the score; prefer concise answers faithful to the passages.
5. Assign the score of the closest level following the rubric above.
6. Provide a brief and clear rationale.

Return only a JSON object: {"rationale": "<1-2 sentences>", "score": <0-1>}
