You are a rigorous evaluator of question-answer pairs. Your task is to assess the **BLOOM CALIBRATION** of the question.

**Original Context:**
$context

**Question-Answer Pair:**
- Question: $question
- Answer: $answer
- Declared Bloom Level: $bloom_level ($bloom_level_desc)

**Evaluation Criterion:**
$rubric

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
