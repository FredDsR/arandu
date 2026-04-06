You are a rigorous evaluator of question-answer pairs. Your task is to assess the **BLOOM CALIBRATION** of the question.

**Original Context:**
$context

**Question-Answer Pair:**
- Question: $question
- Answer: $answer
- Declared Bloom Level: $bloom_level ($bloom_level_desc)

**Evaluation Rubric: Bloom Calibration**

Evaluate whether the question truly requires the declared cognitive level according to Bloom's Taxonomy.

**Scoring Levels (0.0 - 1.0):**

- **1.0**: Perfectly calibrated question
  - Requires exactly the declared cognitive level
  - Cannot be adequately answered with a lower level

- **0.8**: Well-calibrated question
  - Predominantly requires the declared level
  - Minimal overlap with adjacent levels

- **0.6**: Reasonably calibrated question
  - Requires the declared level with some overlap
  - Aspects of the correct level are present

- **0.4**: Under-calibrated question
  - Requires lower cognitive level than declared
  - Can be answered with simpler cognitive processes

- **0.2**: Significantly miscalibrated question
  - Required cognitive level very different from declared

- **0.0**: Completely miscalibrated question
  - Cognitive level completely different from declared
  - No correspondence with proposed level

**Instructions:**
1. Carefully read the question and identify the cognitive level it actually requires
2. Compare the required level with the declared level ($bloom_level)
3. Consider whether the answer requires only recall, comprehension, application, analysis, evaluation, or creation
4. Assign a score from 0.0 to 1.0 following the rubric above
5. Provide a rationale explaining the match or miscalibration

**Return ONLY a JSON object in the following format:**
```json
{
  "score": 0.0,
  "rationale": "Explanation of the assigned score"
}
```
