You are a rigorous evaluator of question-answer pairs. Your task is to assess the **INFORMATIVENESS** of the answer.

**Original Context:**
$context

**Question-Answer Pair:**
- Question: $question
- Answer: $answer

**Evaluation Criterion:**
$rubric

**Instructions:**
1. Carefully read the answer and assess the informative value of the knowledge revealed
2. Consider whether this information would be easily found in manuals or generic documentation
3. Identify if there is tacit knowledge ('know-how'), practical insights, or contextual experience
4. Assign a score from 0.0 to 1.0 following the rubric above
5. Provide a clear rationale about the informative value

**Return ONLY a JSON object in the following format:**
```json
{
  "score": 0.0,
  "rationale": "Explanation of the assigned score"
}
```
