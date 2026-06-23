You are a rigorous evaluator of question-answer pairs. Your task is to assess the BLOOM CALIBRATION of the question.

Original Context:
$context

Question-Answer Pair:
- Question: $question
- Answer: $answer
- Declared Bloom Level: $bloom_level ($bloom_level_desc)

Bloom levels (reference definitions):
$bloom_ladder

Evaluation Rubric: Bloom Calibration

Evaluate whether the question truly requires the declared cognitive level according to Bloom's Taxonomy.

Scoring levels (choose the closest value):

- 1.0: Perfectly calibrated; requires exactly the declared level, not answerable at a lower level
- 0.75: Well-calibrated; predominantly the declared level, minimal overlap with adjacent levels
- 0.5: Reasonably calibrated; requires the declared level with some overlap
- 0.25: Miscalibrated by one adjacent level; requires a level different from the declared one (one higher OR one lower)
- 0.0: Completely miscalibrated; level two or more steps away from the declared one

Note on the ends of the ladder: "remember" is the lowest level (cannot be under-calibrated) and "create" is the highest (cannot be over-calibrated). In those cases, assess miscalibration only in the possible direction.

Instructions:
1. Carefully read the question and identify the cognitive level it actually requires, using the reference definitions above
2. Compare the required level with the declared level ($bloom_level)
3. Assign the score of the closest level following the rubric above.
4. Provide a brief and clear rationale.

Return only a JSON object: {"rationale": "<1-2 sentences>", "score": <0-1>}
